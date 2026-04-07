"""
评测报告生成模块

将评测结果渲染为可读的 Markdown 报告，支持多轮对比。
"""

from pathlib import Path
from typing import Any


class ReportGenerator:
    """将评测结果生成可读的 Markdown 报告"""

    DIMENSION_NAMES = {
        "task_completion": "📋 任务完成度",
        "efficiency": "⚡ 效率指标",
        "collaboration": "🤝 协作质量",
        "tool_effectiveness": "🔧 工具有效性",
        "output_quality": "📝 输出质量",
    }

    SCORE_EMOJI = {
        "excellent": "🟢",  # >= 0.8
        "good": "🟡",  # >= 0.6
        "poor": "🔴",  # < 0.6
    }

    def _score_level(self, score: float) -> str:
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        return "poor"

    def _score_bar(self, score: float, width: int = 20) -> str:
        """生成文本进度条"""
        filled = int(score * width)
        return f"{'█' * filled}{'░' * (width - filled)} {score:.0%}"

    def generate_single_report(self, eval_result: dict, trace: dict | None = None) -> str:
        """生成单次评测的 Markdown 报告"""
        lines = []
        task_id = eval_result.get("task_id", "unknown")
        overall = eval_result.get("overall_score", 0)
        level = self._score_level(overall)
        emoji = self.SCORE_EMOJI[level]

        lines.append(f"# 🎯 Agent 评测报告 — {task_id}")
        lines.append("")
        lines.append(f"## 综合评分: {emoji} {overall:.1%}")
        lines.append(f"```")
        lines.append(f"总分: {self._score_bar(overall)}")
        lines.append(f"```")
        lines.append("")

        # 综合评分计算公式
        dims = eval_result.get("dimensions", [])
        default_weights = {
            "task_completion": 0.30,
            "efficiency": 0.20,
            "collaboration": 0.20,
            "tool_effectiveness": 0.15,
            "output_quality": 0.15,
        }
        if dims:
            lines.append("**加权计算公式：**")
            lines.append("")
            formula_parts = []
            for dim in dims:
                d_name = dim["dimension"]
                d_score = dim.get("score", 0)
                w = default_weights.get(d_name, 0)
                short_name = self.DIMENSION_NAMES.get(d_name, d_name).split(" ", 1)[-1]
                formula_parts.append(f"{short_name}({d_score:.1%}) × {w:.0%}")
            lines.append(f"> {' + '.join(formula_parts)} = **{overall:.1%}**")
            lines.append("")

        # 维度概览表
        lines.append("## 📊 各维度评分")
        lines.append("")
        lines.append("| 维度 | 得分 | 等级 | 说明 |")
        lines.append("|------|------|------|------|")
        for dim in eval_result.get("dimensions", []):
            name = self.DIMENSION_NAMES.get(dim["dimension"], dim["dimension"])
            score = dim.get("score", 0)
            lvl = self._score_level(score)
            em = self.SCORE_EMOJI[lvl]
            details = dim.get("details", "")
            lines.append(f"| {name} | {score:.1%} | {em} | {details} |")
        lines.append("")

        # 各维度详细分析
        lines.append("## 📋 详细分析")
        lines.append("")
        for dim in eval_result.get("dimensions", []):
            name = self.DIMENSION_NAMES.get(dim["dimension"], dim["dimension"])
            score = dim.get("score", 0)
            lines.append(f"### {name} ({score:.1%})")
            lines.append(f"```")
            lines.append(f"{self._score_bar(score)}")
            lines.append(f"```")

            sub_scores = dim.get("sub_scores", {})
            if sub_scores:
                lines.append("**子项评分：**")
                for sub_name, sub_val in sub_scores.items():
                    sub_em = self.SCORE_EMOJI[self._score_level(sub_val)]
                    lines.append(f"- {sub_em} `{sub_name}`: {sub_val:.2f}")
                lines.append("")
                # 得分计算公式
                n = len(sub_scores)
                parts = " + ".join(f"{v:.2f}" for v in sub_scores.values())
                lines.append(f"**得分计算：** ({parts}) / {n} = **{score:.1%}**")
                lines.append(f"> 维度得分 = 各子项算术平均值，每个子项满分 1.0")
            lines.append("")

        # 问题列表
        issues = eval_result.get("issues", [])
        if issues:
            lines.append("## ⚠️ 发现的问题")
            lines.append("")
            for i, issue in enumerate(issues, 1):
                lines.append(f"{i}. {issue}")
            lines.append("")

        # 优化建议
        suggestions = eval_result.get("suggestions", [])
        if suggestions:
            lines.append("## 💡 优化建议")
            lines.append("")
            for i, sug in enumerate(suggestions, 1):
                lines.append(f"{i}. {sug}")
            lines.append("")

        # Trace 统计（可选）
        if trace:
            lines.append("## 📈 运行统计")
            lines.append("")
            messages = trace.get("messages", [])
            agent_counts: dict[str, int] = {}
            for m in messages:
                src = m.get("source", "unknown")
                agent_counts[src] = agent_counts.get(src, 0) + 1

            lines.append(f"- **总消息数**: {len(messages)}")
            lines.append(f"- **停止原因**: {trace.get('stop_reason', 'N/A')}")
            lines.append(f"- **Agent 发言分布**:")
            for agent, count in sorted(agent_counts.items()):
                lines.append(f"  - {agent}: {count} 条")
            lines.append("")

        return "\n".join(lines)

    def generate_comparison_report(self, results: list[dict], labels: list[str] | None = None) -> str:
        """生成多轮评测对比报告"""
        if not results:
            return "# 无评测数据"

        if labels is None:
            labels = [f"Round {i + 1}" for i in range(len(results))]

        lines = []
        lines.append("# 📊 Agent 评测对比报告（多轮调优）")
        lines.append("")

        # 总分趋势
        lines.append("## 🔄 总分趋势")
        lines.append("")
        lines.append("| 轮次 | 总分 | 变化 |")
        lines.append("|------|------|------|")
        prev_score = 0.0
        for i, (r, label) in enumerate(zip(results, labels)):
            score = r.get("overall_score", 0)
            if i == 0:
                delta = "—"
            else:
                d = score - prev_score
                delta = f"{'📈' if d > 0 else '📉' if d < 0 else '➡️'} {d:+.1%}"
            lines.append(f"| {label} | {score:.1%} | {delta} |")
            prev_score = score
        lines.append("")

        # 各维度对比
        all_dims = set()
        for r in results:
            for d in r.get("dimensions", []):
                all_dims.add(d["dimension"])

        lines.append("## 📈 各维度对比")
        lines.append("")

        header = "| 维度 | " + " | ".join(labels) + " |"
        separator = "|------|" + "|".join(["------"] * len(labels)) + "|"
        lines.append(header)
        lines.append(separator)

        for dim_name in sorted(all_dims):
            display_name = self.DIMENSION_NAMES.get(dim_name, dim_name)
            row = f"| {display_name} |"
            for r in results:
                dim_data = next((d for d in r.get("dimensions", []) if d["dimension"] == dim_name), None)
                if dim_data:
                    score = dim_data.get("score", 0)
                    em = self.SCORE_EMOJI[self._score_level(score)]
                    row += f" {em} {score:.1%} |"
                else:
                    row += " — |"
            lines.append(row)
        lines.append("")

        # 改进总结
        if len(results) >= 2:
            first = results[0]
            last = results[-1]
            lines.append("## 🎯 改进总结")
            lines.append("")
            first_score = first.get("overall_score", 0)
            last_score = last.get("overall_score", 0)
            improvement = last_score - first_score
            lines.append(f"- **总分变化**: {first_score:.1%} → {last_score:.1%} "
                          f"({'📈 提升' if improvement > 0 else '📉 下降'} {abs(improvement):.1%})")
            lines.append("")

            # 各维度改进
            for dim_name in sorted(all_dims):
                display_name = self.DIMENSION_NAMES.get(dim_name, dim_name)
                first_dim = next((d for d in first.get("dimensions", []) if d["dimension"] == dim_name), None)
                last_dim = next((d for d in last.get("dimensions", []) if d["dimension"] == dim_name), None)
                if first_dim and last_dim:
                    f_s = first_dim.get("score", 0)
                    l_s = last_dim.get("score", 0)
                    delta = l_s - f_s
                    emoji = "📈" if delta > 0.02 else "📉" if delta < -0.02 else "➡️"
                    lines.append(f"- {display_name}: {f_s:.1%} → {l_s:.1%} ({emoji} {delta:+.1%})")

        return "\n".join(lines)

    def save_report(self, content: str, filepath: str | Path) -> Path:
        """保存报告到文件"""
        fp = Path(filepath)
        fp.parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content)
        return fp
