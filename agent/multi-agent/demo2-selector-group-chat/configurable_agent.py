"""
优化版 SelectorGroupChat — 可配置的 Agent 系统

与原始 selector_group_chat.py 的区别：
- 支持外部传入配置（system_message、selector_prompt 等）
- 支持将运行日志写入文件
- 支持返回结构化的运行结果
"""

import asyncio
import sys
from pathlib import Path
from typing import Sequence

# 将上级目录加入 sys.path，以便引用共享模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console

from model_config import get_model_client


# ====================================================================
# 工具函数（与原始版本一致）
# ====================================================================

def search_tech_info(query: str) -> str:
    """模拟技术信息搜索工具。在实际项目中可接入搜索引擎 API。"""
    knowledge_base = {
        "transformer": (
            "Transformer 架构由 Vaswani 等人在 2017 年的论文 'Attention Is All You Need' 中提出。"
            "它基于自注意力机制（Self-Attention），完全抛弃了 RNN 和 CNN 结构。"
            "核心组件包括：Multi-Head Attention、Position-wise Feed-Forward Network、"
            "Layer Normalization 和 Residual Connection。"
            "参数规模：GPT-3 有 1750 亿参数，GPT-4 预估超过 1 万亿参数。"
        ),
        "moe": (
            "MoE（Mixture of Experts，混合专家模型）是一种条件计算技术。"
            "核心思想是：模型有多个'专家'子网络，但每次推理只激活其中少数几个。"
            "优势：在保持巨大参数量的同时控制计算成本。"
            "代表模型：Switch Transformer（Google）、Mixtral 8x7B（Mistral）、DeepSeek-V2。"
            "DeepSeek-V2 使用了 DeepSeekMoE 架构，236B 总参数但每 token 只激活 21B。"
        ),
        "rlhf": (
            "RLHF（Reinforcement Learning from Human Feedback）是大模型对齐的关键技术。"
            "流程：1) SFT 监督微调 2) 训练奖励模型（Reward Model）3) PPO 策略优化。"
            "替代方案：DPO（Direct Preference Optimization）绕过了奖励模型训练，直接优化策略。"
            "GRPO（Group Relative Policy Optimization）是 DeepSeek 提出的改进方法。"
        ),
    }
    query_lower = query.lower()
    results = []
    for keyword, info in knowledge_base.items():
        if keyword in query_lower:
            results.append(info)
    if results:
        return "\n\n".join(results)
    return f"未找到与 '{query}' 直接相关的信息，请尝试更具体的关键词，如 transformer、moe、rlhf。"


def compare_models(model_a: str, model_b: str) -> str:
    """模拟模型对比分析工具。"""
    comparisons = {
        ("gpt-4", "deepseek"): (
            "对比分析：\n"
            "- GPT-4：闭源，参数量未公开（推测 >1T），训练成本极高\n"
            "- DeepSeek-V2：开源，236B 总参数/21B 激活参数，训练成本约 GPT-4 的 1/10\n"
            "- 性能：在多项基准测试中 DeepSeek-V2 接近 GPT-4 水平\n"
            "- 推理效率：DeepSeek-V2 使用 MoE 架构，推理成本显著更低\n"
            "- 结论：开源模型正在快速追赶闭源模型，MoE 架构是关键突破方向"
        ),
        ("transformer", "mamba"): (
            "对比分析：\n"
            "- Transformer：基于 Self-Attention，O(n²) 复杂度，全局建模能力强\n"
            "- Mamba：基于状态空间模型（SSM），O(n) 复杂度，长序列效率更高\n"
            "- Transformer 在短/中等长度任务上表现优异\n"
            "- Mamba 在超长序列（>100K tokens）处理上有优势\n"
            "- 混合架构（如 Jamba）结合两者优势是未来趋势"
        ),
    }
    key = (model_a.lower(), model_b.lower())
    key_reversed = (model_b.lower(), model_a.lower())
    if key in comparisons:
        return comparisons[key]
    elif key_reversed in comparisons:
        return comparisons[key_reversed]
    return f"暂无 {model_a} 与 {model_b} 的直接对比数据。"


# ====================================================================
# 默认配置
# ====================================================================

DEFAULT_CONFIG = {
    "planning_system_message": (
        "你是一个任务规划专家。你的职责是：\n"
        "1. 将复杂的研究问题分解成可执行的子任务\n"
        "2. 将子任务分配给合适的团队成员\n"
        "3. 在所有子任务完成后汇总结果\n\n"
        "你的团队成员：\n"
        "  - ResearchAgent：擅长信息检索，可以搜索技术资料\n"
        "  - AnalystAgent：擅长数据分析和对比研究\n\n"
        "分配任务时使用格式：\n"
        "  1. <AgentName>: <具体任务描述>\n\n"
        "当所有任务完成、你已汇总结论后，在回复末尾加上 'TERMINATE'。\n"
        "注意：你只负责规划和汇总，不要自己执行搜索或分析任务。"
    ),
    "research_system_message": (
        "你是一个技术研究员。使用 search_tech_info 工具搜索相关信息。\n"
        "每次只执行一个搜索任务，并将搜索结果如实汇报给团队。\n"
        "不要对结果做额外的分析或推测，只负责信息检索。"
    ),
    "analyst_system_message": (
        "你是一个技术分析师。你的职责是：\n"
        "1. 根据 ResearchAgent 收集到的信息进行深度分析\n"
        "2. 使用 compare_models 工具做对比研究\n"
        "3. 给出有洞察力的分析结论\n"
        "基于数据和事实进行分析，不要凭空推测。"
    ),
    "selector_prompt": (
        "根据当前对话上下文，选择最合适的下一个发言智能体。\n\n"
        "可用智能体及其职责：\n{roles}\n\n"
        "当前对话：\n{history}\n\n"
        "请从 {participants} 中选择一个智能体执行下一步任务。\n"
        "规则：\n"
        "- 新任务到来时，应先让 PlanningAgent 规划\n"
        "- PlanningAgent 分配了任务后，选择被指定的执行者\n"
        "- 执行者完成后，如果还有待完成的子任务，选择下一个执行者\n"
        "- 所有子任务完成后，选择 PlanningAgent 汇总结果\n"
        "只选择一个智能体。"
    ),
    "max_messages": 20,
    "model_key": "gpt-5-2",
}

DEFAULT_TASK = (
    "请研究以下问题并给出分析报告：\n"
    "大语言模型中的 MoE（混合专家）架构相比传统 Transformer 有哪些优势？\n"
    "请对比 GPT-4 和 DeepSeek 的技术路线差异。"
)


async def run_with_config(
    config: dict | None = None,
    task: str | None = None,
) -> tuple[str, dict]:
    """使用指定配置运行 SelectorGroupChat。

    Args:
        config: Agent 配置字典，None 则使用默认配置
        task: 任务描述，None 则使用默认任务

    Returns:
        (log_text, run_info) 日志文本和运行元信息
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    task = task or DEFAULT_TASK
    model_client = get_model_client(cfg["model_key"])

    # 创建 Agent
    planning_agent = AssistantAgent(
        name="PlanningAgent",
        description="负责规划任务、分解子任务和汇总最终结果的智能体。",
        model_client=model_client,
        system_message=cfg["planning_system_message"],
    )

    research_agent = AssistantAgent(
        name="ResearchAgent",
        description="负责搜索和检索技术信息的智能体。",
        model_client=model_client,
        tools=[search_tech_info],
        system_message=cfg["research_system_message"],
    )

    analyst_agent = AssistantAgent(
        name="AnalystAgent",
        description="负责数据分析、对比研究和逻辑推理的智能体。",
        model_client=model_client,
        tools=[compare_models],
        system_message=cfg["analyst_system_message"],
    )

    # 终止条件
    text_termination = TextMentionTermination("TERMINATE")
    max_msg_termination = MaxMessageTermination(max_messages=cfg["max_messages"])
    termination = text_termination | max_msg_termination

    # Selector
    def selector_func(
        messages: Sequence[BaseAgentEvent | BaseChatMessage],
    ) -> str | None:
        if messages[-1].source != "PlanningAgent":
            return "PlanningAgent"
        return None

    team = SelectorGroupChat(
        participants=[planning_agent, research_agent, analyst_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=cfg["selector_prompt"],
        selector_func=selector_func,
        allow_repeated_speaker=True,
    )

    # 运行并捕获输出
    # 注意：Console 输出到 stdout，我们通过写文件的方式可靠捕获
    result = await Console(team.run_stream(task=task))

    # 从 result.messages 重建日志文本
    log_lines = []
    for msg in result.messages:
        source = getattr(msg, "source", "unknown")
        msg_type = type(msg).__name__
        content = getattr(msg, "content", "")

        log_lines.append(f"---------- {msg_type} ({source}) ----------")

        # 处理工具调用类消息
        if hasattr(msg, "content") and isinstance(content, list):
            for item in content:
                log_lines.append(str(item))
        elif hasattr(msg, "content"):
            log_lines.append(str(content))

    stop_str = str(result.stop_reason) if result.stop_reason else "unknown"
    log_lines.append(f"\n✅ 任务完成！停止原因: {stop_str}")
    log_lines.append(f"   共产生 {len(result.messages)} 条消息")

    log_text = "\n".join(log_lines)

    run_info = {
        "stop_reason": stop_str,
        "total_messages": len(result.messages),
        "config": cfg,
        "task": task,
    }

    return log_text, run_info


if __name__ == "__main__":
    log, info = asyncio.run(run_with_config(capture_output=False))
    print(f"\n停止原因: {info['stop_reason']}")
    print(f"消息数: {info['total_messages']}")
