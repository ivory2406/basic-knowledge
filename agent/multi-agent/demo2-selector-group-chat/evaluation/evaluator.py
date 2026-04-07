"""
Agent 评测模块 (Phase 2 & 4: Evaluation)

多维度评测 Agent 系统表现：
1. 任务完成度 (Task Completion)
2. 效率指标 (Efficiency)
3. 协作质量 (Collaboration Quality)
4. 工具使用有效性 (Tool Effectiveness)
5. 输出质量 (Output Quality) — 基于 LLM-as-Judge
"""

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any
from pathlib import Path


@dataclass
class DimensionScore:
    """单维度评分"""
    dimension: str
    score: float = 0.0  # 0.0 ~ 1.0
    max_score: float = 1.0
    details: str = ""
    sub_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """完整评测结果"""
    task_id: str
    overall_score: float = 0.0
    dimensions: list[DimensionScore] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)  # 发现的问题
    suggestions: list[str] = field(default_factory=list)  # 优化建议
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_overall(self, weights: dict[str, float] | None = None):
        """加权计算总分"""
        default_weights = {
            "task_completion": 0.30,
            "efficiency": 0.20,
            "collaboration": 0.20,
            "tool_effectiveness": 0.15,
            "output_quality": 0.15,
        }
        w = weights or default_weights
        total_weight = 0.0
        weighted_sum = 0.0
        for dim in self.dimensions:
            if dim.dimension in w:
                weighted_sum += dim.score * w[dim.dimension]
                total_weight += w[dim.dimension]
        self.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0


class AgentEvaluator:
    """Agent 系统多维度评测器"""

    def __init__(self, model_client=None):
        self.model_client = model_client  # 可选，用于 LLM-as-Judge

    def evaluate(self, trace: dict) -> EvaluationResult:
        """对一次运行 Trace 进行全面评测"""
        result = EvaluationResult(task_id=trace.get("task_id", "unknown"))

        # 逐维度评测
        result.dimensions.append(self._eval_task_completion(trace))
        result.dimensions.append(self._eval_efficiency(trace))
        result.dimensions.append(self._eval_collaboration(trace))
        result.dimensions.append(self._eval_tool_effectiveness(trace))
        result.dimensions.append(self._eval_output_quality(trace))

        # 计算总分
        result.compute_overall()

        # 汇总问题和建议
        for dim in result.dimensions:
            if dim.score < 0.6:
                result.issues.append(f"[{dim.dimension}] 得分偏低({dim.score:.2f}): {dim.details}")

        result.suggestions = self._generate_suggestions(result)
        return result

    # ------------------------------------------------------------------
    # 维度 1: 任务完成度
    # ------------------------------------------------------------------
    def _eval_task_completion(self, trace: dict) -> DimensionScore:
        """评估任务是否成功完成"""
        score = DimensionScore(dimension="task_completion", details="")
        sub = {}

        # 1.1 是否正常终止（含 TERMINATE）
        stop_reason = trace.get("stop_reason", "")
        if "TERMINATE" in stop_reason.upper() or "terminate" in stop_reason.lower() or "Text" in stop_reason:
            sub["normal_termination"] = 1.0
        elif "MaxMessage" in stop_reason:
            sub["normal_termination"] = 0.3  # 到达上限才停止
        else:
            sub["normal_termination"] = 0.5

        # 1.2 最终输出是否包含分析报告结构
        messages = trace.get("messages", [])
        final_messages = [m for m in messages if m.get("source") == "PlanningAgent"]
        if final_messages:
            last_planner_msg = final_messages[-1].get("full_content", "")
            # 检查是否有结构化输出
            has_sections = len(re.findall(r"#{1,3}\s", last_planner_msg)) >= 2
            has_conclusion = any(kw in last_planner_msg for kw in ["结论", "总结", "汇总", "TERMINATE"])
            sub["structured_output"] = 1.0 if has_sections else 0.4
            sub["has_conclusion"] = 1.0 if has_conclusion else 0.3
        else:
            sub["structured_output"] = 0.0
            sub["has_conclusion"] = 0.0

        # 1.3 任务要求覆盖度：检查是否涵盖了任务输入中的关键词
        task_input = trace.get("task_input", "")
        all_content = " ".join(m.get("full_content", "") for m in messages)
        key_topics = self._extract_key_topics(task_input)
        covered = sum(1 for t in key_topics if t.lower() in all_content.lower())
        sub["topic_coverage"] = covered / max(len(key_topics), 1)

        score.sub_scores = sub
        score.score = sum(sub.values()) / max(len(sub), 1)
        score.details = f"正常终止={sub.get('normal_termination', 0):.1f}, " \
                        f"结构化输出={sub.get('structured_output', 0):.1f}, " \
                        f"结论总结={sub.get('has_conclusion', 0):.1f}, " \
                        f"话题覆盖={sub.get('topic_coverage', 0):.1f}"
        return score

    # ------------------------------------------------------------------
    # 维度 2: 效率指标
    # ------------------------------------------------------------------
    def _eval_efficiency(self, trace: dict) -> DimensionScore:
        """评估交互效率"""
        score = DimensionScore(dimension="efficiency", details="")
        sub = {}
        messages = trace.get("messages", [])
        total = len(messages)

        # 2.1 消息轮数效率（越少越好，20条上限，理想 5-10 条）
        if total <= 8:
            sub["turn_efficiency"] = 1.0
        elif total <= 12:
            sub["turn_efficiency"] = 0.8
        elif total <= 16:
            sub["turn_efficiency"] = 0.5
        else:
            sub["turn_efficiency"] = 0.3

        # 2.2 无效轮次（重复内容、空响应、工具调用失败等）
        invalid_turns = 0
        for m in messages:
            content = m.get("full_content", "").strip()
            if not content:
                invalid_turns += 1
            elif "暂无" in content and "直接对比数据" in content:
                invalid_turns += 1  # 工具无法回答
            elif "未找到" in content and len(content) < 100:
                invalid_turns += 1
        sub["valid_turn_ratio"] = 1.0 - (invalid_turns / max(total, 1))

        # 2.3 PlanningAgent 的调度效率（不重复分配已完成任务）
        planner_msgs = [m for m in messages if m.get("source") == "PlanningAgent"]
        repeated_assignments = 0
        seen_assignments: set[str] = set()
        for pm in planner_msgs:
            content = pm.get("full_content", "")
            # 检查是否有重复分配
            assignments = re.findall(r"(ResearchAgent|AnalystAgent):\s*(.{10,50})", content)
            for agent, task_desc in assignments:
                key = f"{agent}:{task_desc[:20]}"
                if key in seen_assignments:
                    repeated_assignments += 1
                seen_assignments.add(key)
        sub["scheduling_efficiency"] = 1.0 if repeated_assignments == 0 else max(0.3, 1.0 - repeated_assignments * 0.2)

        score.sub_scores = sub
        score.score = sum(sub.values()) / max(len(sub), 1)
        score.details = f"轮次效率={sub.get('turn_efficiency', 0):.1f}, " \
                        f"有效率={sub.get('valid_turn_ratio', 0):.2f}, " \
                        f"调度效率={sub.get('scheduling_efficiency', 0):.1f}"
        return score

    # ------------------------------------------------------------------
    # 维度 3: 协作质量
    # ------------------------------------------------------------------
    def _eval_collaboration(self, trace: dict) -> DimensionScore:
        """评估多 Agent 协作质量"""
        score = DimensionScore(dimension="collaboration", details="")
        sub = {}
        messages = trace.get("messages", [])

        # 3.1 Agent 参与度分布（理想：三个 Agent 都有贡献）
        agent_counts: dict[str, int] = {}
        for m in messages:
            src = m.get("source", "unknown")
            agent_counts[src] = agent_counts.get(src, 0) + 1

        expected_agents = {"PlanningAgent", "ResearchAgent", "AnalystAgent"}
        participated = expected_agents & set(agent_counts.keys())
        sub["participation_rate"] = len(participated) / len(expected_agents)

        # 3.2 角色分工是否合理（Planner 规划/汇总、Researcher 搜索、Analyst 分析）
        role_adherence = 0.0
        # Planner 应该有"规划"和"汇总"行为
        planner_msgs = [m.get("full_content", "") for m in messages if m.get("source") == "PlanningAgent"]
        if planner_msgs:
            has_plan = any(any(kw in msg for kw in ["任务", "分配", "检索", "分析"]) for msg in planner_msgs)
            has_summary = any(any(kw in msg for kw in ["汇总", "报告", "结论", "TERMINATE"]) for msg in planner_msgs)
            role_adherence += (0.5 if has_plan else 0.0) + (0.5 if has_summary else 0.0)
        sub["role_adherence"] = role_adherence

        # 3.3 信息传递有效性（后续 Agent 是否利用了前面的信息）
        researcher_content = " ".join(
            m.get("full_content", "") for m in messages if m.get("source") == "ResearchAgent"
        )
        analyst_content = " ".join(
            m.get("full_content", "") for m in messages if m.get("source") == "AnalystAgent"
        )
        planner_final = planner_msgs[-1] if planner_msgs else ""

        # 检查 Planner 的汇总是否引用了 Researcher/Analyst 的内容
        if researcher_content and planner_final:
            # 提取研究者关键词
            r_keywords = set(re.findall(r"[\u4e00-\u9fff]{2,4}", researcher_content))
            p_keywords = set(re.findall(r"[\u4e00-\u9fff]{2,4}", planner_final))
            overlap = len(r_keywords & p_keywords)
            sub["info_integration"] = min(1.0, overlap / max(10, len(r_keywords) * 0.3))
        else:
            sub["info_integration"] = 0.3

        # 3.4 对话流畅度（无死循环、无冲突）
        # 检查是否有连续重复内容
        consecutive_repeats = 0
        for i in range(1, len(messages)):
            if messages[i].get("content", "")[:50] == messages[i - 1].get("content", "")[:50]:
                consecutive_repeats += 1
        sub["flow_quality"] = 1.0 if consecutive_repeats == 0 else max(0.3, 1.0 - consecutive_repeats * 0.2)

        score.sub_scores = sub
        score.score = sum(sub.values()) / max(len(sub), 1)
        score.details = f"参与度={sub.get('participation_rate', 0):.1f}, " \
                        f"角色遵循={sub.get('role_adherence', 0):.1f}, " \
                        f"信息整合={sub.get('info_integration', 0):.2f}"
        return score

    # ------------------------------------------------------------------
    # 维度 4: 工具使用有效性
    # ------------------------------------------------------------------
    def _eval_tool_effectiveness(self, trace: dict) -> DimensionScore:
        """评估工具调用的有效性"""
        score = DimensionScore(dimension="tool_effectiveness", details="")
        sub = {}
        messages = trace.get("messages", [])

        # 统计所有工具调用
        all_tool_calls: list[dict] = []
        for m in messages:
            for tc in m.get("tool_calls", []):
                all_tool_calls.append(tc)

        total_calls = len(all_tool_calls)
        if total_calls == 0:
            score.score = 0.5
            score.details = "无工具调用"
            return score

        # 4.1 工具调用成功率
        errors = sum(1 for tc in all_tool_calls if tc.get("is_error", False))
        sub["success_rate"] = 1.0 - (errors / total_calls)

        # 4.2 工具结果有效性（结果是否被后续利用）
        useful_results = 0
        for tc in all_tool_calls:
            result = tc.get("result", "")
            if result and "未找到" not in result and "暂无" not in result:
                useful_results += 1
        sub["result_usefulness"] = useful_results / total_calls

        # 4.3 工具选择合理性（是否选对了工具）
        # ResearchAgent 应调用 search_tech_info，AnalystAgent 应调用 compare_models
        correct_tool = 0
        for tc in all_tool_calls:
            agent = tc.get("agent", "")
            tool = tc.get("tool_name", "")
            if (agent == "ResearchAgent" and tool == "search_tech_info") or \
               (agent == "AnalystAgent" and tool == "compare_models"):
                correct_tool += 1
        sub["tool_selection"] = correct_tool / total_calls if total_calls > 0 else 0.5

        # 4.4 参数质量（是否使用了有效关键词）
        good_params = 0
        for tc in all_tool_calls:
            args = tc.get("arguments", "")
            # 检查是否使用了模拟知识库中的关键词
            if any(kw in args.lower() for kw in ["transformer", "moe", "rlhf", "gpt", "deepseek"]):
                good_params += 1
        sub["param_quality"] = good_params / total_calls if total_calls > 0 else 0.0

        score.sub_scores = sub
        score.score = sum(sub.values()) / max(len(sub), 1)
        score.details = f"成功率={sub.get('success_rate', 0):.2f}, " \
                        f"结果有效={sub.get('result_usefulness', 0):.2f}, " \
                        f"工具选择={sub.get('tool_selection', 0):.2f}, " \
                        f"参数质量={sub.get('param_quality', 0):.2f}"
        return score

    # ------------------------------------------------------------------
    # 维度 5: 输出质量
    # ------------------------------------------------------------------
    def _eval_output_quality(self, trace: dict) -> DimensionScore:
        """评估最终输出的质量"""
        score = DimensionScore(dimension="output_quality", details="")
        sub = {}
        messages = trace.get("messages", [])

        # 找到最终的汇总报告（Planner 最后一条含有 TERMINATE 的消息）
        final_report = ""
        for m in reversed(messages):
            if m.get("source") == "PlanningAgent" and "TERMINATE" in m.get("full_content", ""):
                final_report = m.get("full_content", "")
                break

        if not final_report:
            # 取 Planner 的最后一条消息
            planner_msgs = [m for m in messages if m.get("source") == "PlanningAgent"]
            if planner_msgs:
                final_report = planner_msgs[-1].get("full_content", "")

        if not final_report:
            score.score = 0.0
            score.details = "未找到最终报告"
            return score

        # 5.1 长度充实度（太短缺乏信息，太长可能冗余）
        char_count = len(final_report)
        if 500 <= char_count <= 3000:
            sub["length_adequacy"] = 1.0
        elif 200 <= char_count < 500 or 3000 < char_count <= 5000:
            sub["length_adequacy"] = 0.7
        else:
            sub["length_adequacy"] = 0.4

        # 5.2 结构性（有标题、段落、列表）
        has_headers = len(re.findall(r"#{1,3}\s", final_report)) >= 2
        has_list = bool(re.search(r"^\s*[-*\d.]\s", final_report, re.MULTILINE))
        sub["structure"] = (0.5 if has_headers else 0.0) + (0.5 if has_list else 0.0)

        # 5.3 信息密度（包含技术术语的比例）
        tech_terms = ["MoE", "Transformer", "GPT", "DeepSeek", "参数", "推理", "训练",
                      "注意力", "专家", "稀疏", "路由", "对齐", "RLHF", "架构"]
        term_count = sum(1 for t in tech_terms if t.lower() in final_report.lower())
        sub["info_density"] = min(1.0, term_count / 8)

        # 5.4 客观性标注（是否区分了"已知/推测/未知"）
        has_annotation = any(kw in final_report for kw in ["推测", "未知", "公开", "未公开", "可能"])
        sub["objectivity"] = 1.0 if has_annotation else 0.4

        score.sub_scores = sub
        score.score = sum(sub.values()) / max(len(sub), 1)
        score.details = f"长度={sub.get('length_adequacy', 0):.1f}, " \
                        f"结构={sub.get('structure', 0):.1f}, " \
                        f"信息密度={sub.get('info_density', 0):.2f}, " \
                        f"客观性={sub.get('objectivity', 0):.1f}"
        return score

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _extract_key_topics(self, task_input: str) -> list[str]:
        """从任务输入中提取关键话题"""
        topics = []
        keywords = ["MoE", "混合专家", "Transformer", "GPT-4", "DeepSeek", "优势", "对比", "技术路线"]
        for kw in keywords:
            if kw.lower() in task_input.lower():
                topics.append(kw)
        return topics if topics else ["MoE", "Transformer", "GPT-4", "DeepSeek"]

    def _generate_suggestions(self, result: EvaluationResult) -> list[str]:
        """基于评测结果生成优化建议"""
        suggestions = []

        for dim in result.dimensions:
            if dim.dimension == "task_completion" and dim.score < 0.8:
                suggestions.append(
                    "[任务完成] 增强 PlanningAgent 的汇总能力：在 system_message 中要求必须输出"
                    "结构化报告（含标题、要点、结论），并在分配任务时更明确地列出预期输出格式。"
                )
            if dim.dimension == "efficiency" and dim.score < 0.7:
                if dim.sub_scores.get("turn_efficiency", 1) < 0.8:
                    suggestions.append(
                        "[效率] 减少无效轮次：1) 优化 selector_func，让 PlanningAgent 在检查进度时"
                        "更快决定是否需要补充信息；2) 提高 MaxMessageTermination 的阈值或加入"
                        "更精准的终止条件。"
                    )
                if dim.sub_scores.get("valid_turn_ratio", 1) < 0.8:
                    suggestions.append(
                        "[效率] 减少无效消息：工具返回'未找到'时，让 Agent 用备选关键词重试"
                        "而非直接将空结果汇报。"
                    )
            if dim.dimension == "collaboration" and dim.score < 0.7:
                suggestions.append(
                    "[协作] 增强信息传递：1) 在 selector_prompt 中强调后续 Agent 必须利用"
                    "前面收集的信息；2) PlanningAgent 的 system_message 中要求汇总时引用"
                    "ResearchAgent 和 AnalystAgent 的关键发现。"
                )
            if dim.dimension == "tool_effectiveness" and dim.score < 0.7:
                suggestions.append(
                    "[工具] 提升工具使用效果：1) 优化 ResearchAgent 的 system_message，"
                    "要求使用模拟知识库中的关键词（transformer/moe/rlhf）进行精准搜索；"
                    "2) AnalystAgent 调用 compare_models 时参数应精简为知识库支持的格式。"
                )
            if dim.dimension == "output_quality" and dim.score < 0.7:
                suggestions.append(
                    "[输出] 提升报告质量：在 PlanningAgent 的 system_message 中明确要求"
                    "最终报告必须包含：① 结构化标题 ② 关键技术术语 ③ 对比表格/列表 "
                    "④ 区分'已公开/推测/未知'。"
                )

        if not suggestions:
            suggestions.append("当前各维度表现良好，可尝试增加任务复杂度进行压力测试。")

        return suggestions

    def save_result(self, result: EvaluationResult, output_dir: str = "evaluation/results") -> Path:
        """保存评测结果"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / f"{result.task_id}_eval.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        return filepath
