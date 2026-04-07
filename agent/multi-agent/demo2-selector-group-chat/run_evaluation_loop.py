"""
🔄 闭环评测主驱动脚本 (Closed-Loop Evaluation Pipeline)

完整流程：
  数据采集 → Agent评测 → Agent优化 → 再次运行 → 再次评测 → 对比报告

Usage:
    cd agent/multi-agent/demo2-selector-group-chat
    python run_evaluation_loop.py
"""

import asyncio
import json
import sys
import warnings
from pathlib import Path
from dataclasses import asdict

# 抑制 autogen 的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# 确保能找到上级共享模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.data_collector import DataCollector
from evaluation.evaluator import AgentEvaluator
from evaluation.optimizer import AgentOptimizer
from evaluation.report_generator import ReportGenerator
from configurable_agent import run_with_config, DEFAULT_CONFIG, DEFAULT_TASK


def print_banner(title: str, char: str = "=", width: int = 70):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}\n")


async def run_evaluation_loop(max_rounds: int = 2):
    """运行闭环评测流程

    Args:
        max_rounds: 最大优化轮次（含初始轮）
    """
    collector = DataCollector(output_dir="evaluation/traces")
    evaluator = AgentEvaluator()
    optimizer = AgentOptimizer()
    reporter = ReportGenerator()

    all_eval_results: list[dict] = []
    all_traces: list[dict] = []
    current_config = dict(DEFAULT_CONFIG)
    labels: list[str] = []

    for round_idx in range(max_rounds):
        is_baseline = (round_idx == 0)
        round_label = "Baseline（基线）" if is_baseline else f"Round {round_idx}（优化后）"
        labels.append(round_label)

        # ==============================================================
        # Phase 1: 数据采集 — 运行 Agent
        # ==============================================================
        print_banner(f"📡 Phase 1: 数据采集 — {round_label}", "─")
        print(f"运行配置变更项: {list(set(current_config.keys()) - set(DEFAULT_CONFIG.keys())) if round_idx > 0 else '(默认配置)'}")

        task_id = f"task_{round_idx:03d}"

        try:
            log_text, run_info = await run_with_config(
                config=current_config,
                task=DEFAULT_TASK,
            )
        except Exception as e:
            print(f"❌ 运行失败: {e}")
            import traceback
            traceback.print_exc()
            break

        print(f"✅ 运行完成: {run_info['total_messages']} 条消息, 停止原因: {run_info['stop_reason']}")

        # 保存运行日志
        log_path = Path("evaluation/logs") / f"{task_id}_output.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(log_text)
        print(f"📝 日志保存: {log_path}")

        # 解析 Trace
        trace = collector.parse_run_output(
            log_text=log_text,
            task_id=task_id,
            task_input=DEFAULT_TASK,
            config=current_config,
        )
        trace_path = collector.save_trace(trace)
        trace_dict = asdict(trace)
        all_traces.append(trace_dict)
        print(f"📦 Trace 保存: {trace_path}")

        # ==============================================================
        # Phase 2: Agent 评测
        # ==============================================================
        print_banner(f"🔍 Phase 2: Agent 评测 — {round_label}", "─")

        eval_result = evaluator.evaluate(trace_dict)
        eval_dict = asdict(eval_result)
        all_eval_results.append(eval_dict)

        # 保存评测结果
        eval_path = evaluator.save_result(eval_result)
        print(f"📊 评测结果保存: {eval_path}")

        # 打印评分概览
        print(f"\n{'─' * 50}")
        print(f"  综合评分: {eval_result.overall_score:.1%}")
        print(f"{'─' * 50}")
        for dim in eval_result.dimensions:
            emoji = "🟢" if dim.score >= 0.8 else "🟡" if dim.score >= 0.6 else "🔴"
            print(f"  {emoji} {dim.dimension:25s} {dim.score:.1%}  ({dim.details})")

        if eval_result.issues:
            print(f"\n  ⚠️  问题: {len(eval_result.issues)} 项")
            for issue in eval_result.issues:
                print(f"    • {issue[:80]}...")

        # 生成单次报告
        single_report = reporter.generate_single_report(eval_dict, trace_dict)
        report_path = reporter.save_report(
            single_report,
            f"evaluation/reports/{task_id}_report.md"
        )
        print(f"📋 单轮报告: {report_path}")

        # ==============================================================
        # Phase 3: Agent 优化（非最后一轮时执行）
        # ==============================================================
        if round_idx < max_rounds - 1:
            print_banner(f"🔧 Phase 3: Agent 优化 — 生成优化方案", "─")

            opt_plan = optimizer.generate_plan(eval_dict, current_config)

            if opt_plan.actions:
                print(f"📋 生成 {len(opt_plan.actions)} 项优化动作:")
                for i, action in enumerate(opt_plan.actions, 1):
                    print(f"  {i}. [{action.target}] {action.description}")
                    print(f"     预期影响: {action.expected_impact}")

                # 应用优化
                current_config = opt_plan.config_after
                plan_path = optimizer.save_plan(opt_plan)
                print(f"\n✅ 优化方案已应用，保存: {plan_path}")

                # 显示配置变更
                changed_keys = [
                    k for k in current_config
                    if current_config.get(k) != DEFAULT_CONFIG.get(k)
                ]
                if changed_keys:
                    print(f"📝 变更的配置项: {changed_keys}")
            else:
                print("✅ 各维度表现良好，无需优化。")
                break  # 已经足够好了

        print()

    # ==================================================================
    # Phase 4: 生成对比报告
    # ==================================================================
    print_banner("📊 Phase 4: 多轮对比报告", "=")

    comparison_report = reporter.generate_comparison_report(all_eval_results, labels)
    comparison_path = reporter.save_report(
        comparison_report,
        "evaluation/reports/comparison_report.md"
    )
    print(f"📊 对比报告: {comparison_path}")

    # 打印最终对比
    if len(all_eval_results) >= 2:
        first = all_eval_results[0]
        last = all_eval_results[-1]
        improvement = last["overall_score"] - first["overall_score"]
        emoji = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
        print(f"\n{'=' * 50}")
        print(f"  🎯 最终结果: {first['overall_score']:.1%} → {last['overall_score']:.1%} ({emoji} {improvement:+.1%})")
        print(f"{'=' * 50}")

        for dim_name in ["task_completion", "efficiency", "collaboration", "tool_effectiveness", "output_quality"]:
            first_dim = next((d for d in first["dimensions"] if d["dimension"] == dim_name), None)
            last_dim = next((d for d in last["dimensions"] if d["dimension"] == dim_name), None)
            if first_dim and last_dim:
                f_s = first_dim["score"]
                l_s = last_dim["score"]
                d = l_s - f_s
                de = "📈" if d > 0.02 else "📉" if d < -0.02 else "➡️"
                print(f"  {de} {dim_name:25s} {f_s:.1%} → {l_s:.1%} ({d:+.1%})")

    print(f"\n✅ 闭环评测完成！所有报告保存在 evaluation/reports/ 目录")
    print(f"   - 单轮报告: evaluation/reports/task_XXX_report.md")
    print(f"   - 对比报告: evaluation/reports/comparison_report.md")
    print(f"   - Trace 数据: evaluation/traces/")
    print(f"   - 优化方案: evaluation/plans/")


if __name__ == "__main__":
    asyncio.run(run_evaluation_loop(max_rounds=2))
