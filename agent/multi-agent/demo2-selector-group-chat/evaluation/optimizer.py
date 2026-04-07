"""
Agent 优化模块 (Phase 3: Optimization)

基于评测结果自动生成优化方案，并应用到 Agent 配置中。
优化维度：
1. System Message 优化（增强指令清晰度）
2. Selector 策略优化（调度逻辑改进）
3. 工具配置优化（参数引导、重试策略）
4. 终止条件优化
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OptimizationAction:
    """单个优化动作"""
    target: str  # 优化目标（如 "PlanningAgent.system_message"）
    action_type: str  # 动作类型：patch_prompt / adjust_config / add_constraint
    description: str  # 优化描述
    before: str  # 修改前（摘要）
    after: str  # 修改后（摘要）
    expected_impact: str  # 预期影响


@dataclass
class OptimizationPlan:
    """优化方案"""
    plan_id: str
    source_eval_id: str  # 基于哪次评测
    actions: list[OptimizationAction] = field(default_factory=list)
    config_before: dict = field(default_factory=dict)
    config_after: dict = field(default_factory=dict)


class AgentOptimizer:
    """基于评测结果的 Agent 自动优化器"""

    # 优化后的 Prompt 模板库
    PROMPT_ENHANCEMENTS = {
        "planning_structured_output": (
            "\n\n【报告格式要求】\n"
            "最终报告必须采用以下格式：\n"
            "## 一、<主题1标题>\n"
            "- 要点 1\n"
            "- 要点 2\n"
            "## 二、<主题2标题>\n"
            "...\n"
            "## 结论\n"
            "简要汇总核心发现。\n"
            "对于信息来源，请明确标注[已公开]/[推测]/[未知]。"
        ),
        "planning_efficient_check": (
            "\n\n【效率要求】\n"
            "检查进度时，如果信息已足够回答任务问题，直接进入汇总阶段。\n"
            "不要重复分配已完成的子任务。\n"
            "目标：在 10 条消息内完成整个任务。"
        ),
        "research_retry_strategy": (
            "\n\n【搜索策略】\n"
            "搜索时请使用精准的技术关键词。如果搜索结果不理想，请：\n"
            "1. 尝试拆分关键词重新搜索（如将 'MoE优势' 拆为先搜 'moe'，再搜 'transformer'）\n"
            "2. 尝试英文关键词（如 transformer, moe, rlhf）\n"
            "确保搜索结果包含具体的技术细节，而非空泛描述。"
        ),
        "analyst_precise_params": (
            "\n\n【工具使用规范】\n"
            "调用 compare_models 时，参数应使用简洁的模型名称（如 'gpt-4'、'deepseek'），"
            "而非完整名称或带版本号的名称。\n"
            "分析结论必须基于工具返回的数据，如果数据不足，应明确说明信息缺口。"
        ),
        "selector_prompt_enhanced": (
            "根据当前对话上下文，选择最合适的下一个发言智能体。\n\n"
            "可用智能体及其职责：\n{roles}\n\n"
            "当前对话：\n{history}\n\n"
            "请从 {participants} 中选择一个智能体执行下一步任务。\n"
            "规则：\n"
            "- 新任务到来时，应先让 PlanningAgent 规划\n"
            "- PlanningAgent 分配了任务后，选择被指定的执行者\n"
            "- 执行者完成后，如果还有待完成的子任务，选择下一个执行者\n"
            "- 所有子任务完成后，选择 PlanningAgent 汇总结果\n"
            "- 如果某个 Agent 的工具返回了无用结果（如'未找到'），让 PlanningAgent 决定是重试还是跳过\n"
            "- 优先推进未完成的子任务，避免反复在同一问题上打转\n"
            "只选择一个智能体。"
        ),
    }

    def generate_plan(self, eval_result: dict, current_config: dict) -> OptimizationPlan:
        """基于评测结果生成优化方案"""
        plan = OptimizationPlan(
            plan_id=f"opt_{eval_result.get('task_id', 'unknown')}",
            source_eval_id=eval_result.get("task_id", "unknown"),
            config_before=copy.deepcopy(current_config),
        )

        dimensions = {d["dimension"]: d for d in eval_result.get("dimensions", [])}

        # 策略 1: 任务完成度低 → 增强 PlanningAgent 的输出格式要求
        tc = dimensions.get("task_completion", {})
        if tc.get("score", 1.0) < 0.85:
            plan.actions.append(OptimizationAction(
                target="PlanningAgent.system_message",
                action_type="patch_prompt",
                description="增加结构化报告格式要求，提升任务完成度",
                before="原始 system_message（无格式要求）",
                after="追加【报告格式要求】段落",
                expected_impact="预期 task_completion +0.1~0.2",
            ))

        # 策略 2: 效率低 → 增加效率约束
        eff = dimensions.get("efficiency", {})
        if eff.get("score", 1.0) < 0.75:
            plan.actions.append(OptimizationAction(
                target="PlanningAgent.system_message",
                action_type="patch_prompt",
                description="增加效率约束，减少无效轮次",
                before="原始 system_message（无效率约束）",
                after="追加【效率要求】段落",
                expected_impact="预期 efficiency +0.1~0.2，消息数减少 2-4 条",
            ))

        # 策略 3: 工具使用差 → 优化 ResearchAgent 的搜索策略
        tool = dimensions.get("tool_effectiveness", {})
        if tool.get("score", 1.0) < 0.75:
            tool_sub = tool.get("sub_scores", {})
            if tool_sub.get("result_usefulness", 1.0) < 0.7:
                plan.actions.append(OptimizationAction(
                    target="ResearchAgent.system_message",
                    action_type="patch_prompt",
                    description="增加搜索重试策略，提升工具结果有效性",
                    before="原始 system_message（无重试策略）",
                    after="追加【搜索策略】段落",
                    expected_impact="预期 tool_effectiveness.result_usefulness +0.2",
                ))
            if tool_sub.get("param_quality", 1.0) < 0.7 or tool_sub.get("tool_selection", 1.0) < 0.8:
                plan.actions.append(OptimizationAction(
                    target="AnalystAgent.system_message",
                    action_type="patch_prompt",
                    description="规范 AnalystAgent 的工具参数格式",
                    before="原始 system_message（无参数规范）",
                    after="追加【工具使用规范】段落",
                    expected_impact="预期 tool_effectiveness.param_quality +0.2",
                ))

        # 策略 4: 协作质量差 → 优化 selector_prompt
        collab = dimensions.get("collaboration", {})
        if collab.get("score", 1.0) < 0.75:
            plan.actions.append(OptimizationAction(
                target="selector_prompt",
                action_type="patch_prompt",
                description="增强 Selector Prompt，改善调度逻辑",
                before="原始 selector_prompt",
                after="增加无用结果处理规则和效率优先原则",
                expected_impact="预期 collaboration +0.1",
            ))

        # 策略 5: 输出质量差 → 综合增强
        oq = dimensions.get("output_quality", {})
        if oq.get("score", 1.0) < 0.75:
            oq_sub = oq.get("sub_scores", {})
            if oq_sub.get("objectivity", 1.0) < 0.7:
                plan.actions.append(OptimizationAction(
                    target="PlanningAgent.system_message",
                    action_type="patch_prompt",
                    description="要求报告区分信息可信度（已公开/推测/未知）",
                    before="原始 system_message",
                    after="追加可信度标注要求",
                    expected_impact="预期 output_quality.objectivity +0.3",
                ))

        plan.config_after = self.apply_plan(plan, current_config)
        return plan

    def apply_plan(self, plan: OptimizationPlan, config: dict) -> dict:
        """将优化方案应用到配置中，返回新配置"""
        new_config = copy.deepcopy(config)

        for action in plan.actions:
            if action.action_type == "patch_prompt":
                target = action.target

                if target == "PlanningAgent.system_message":
                    current = new_config.get("planning_system_message", "")
                    if "报告格式要求" in action.description:
                        current += self.PROMPT_ENHANCEMENTS["planning_structured_output"]
                    if "效率约束" in action.description:
                        current += self.PROMPT_ENHANCEMENTS["planning_efficient_check"]
                    if "可信度标注" in action.description and "公开" not in current:
                        current += "\n请在分析中标注信息的可信度：已公开/推测/未知。"
                    new_config["planning_system_message"] = current

                elif target == "ResearchAgent.system_message":
                    current = new_config.get("research_system_message", "")
                    current += self.PROMPT_ENHANCEMENTS["research_retry_strategy"]
                    new_config["research_system_message"] = current

                elif target == "AnalystAgent.system_message":
                    current = new_config.get("analyst_system_message", "")
                    current += self.PROMPT_ENHANCEMENTS["analyst_precise_params"]
                    new_config["analyst_system_message"] = current

                elif target == "selector_prompt":
                    new_config["selector_prompt"] = self.PROMPT_ENHANCEMENTS["selector_prompt_enhanced"]

        return new_config

    def save_plan(self, plan: OptimizationPlan, output_dir: str = "evaluation/plans") -> Path:
        """保存优化方案"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / f"{plan.plan_id}_plan.json"

        data = {
            "plan_id": plan.plan_id,
            "source_eval_id": plan.source_eval_id,
            "actions": [
                {
                    "target": a.target,
                    "action_type": a.action_type,
                    "description": a.description,
                    "before": a.before,
                    "after": a.after,
                    "expected_impact": a.expected_impact,
                }
                for a in plan.actions
            ],
            "config_diff_keys": list(
                set(plan.config_after.keys()) - set(plan.config_before.keys())
                | {k for k in plan.config_after if plan.config_after.get(k) != plan.config_before.get(k)}
            ),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath
