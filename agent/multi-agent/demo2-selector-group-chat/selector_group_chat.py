"""
SelectorGroupChat 示例 —— 智能任务调度模式

场景：研究一个技术问题并给出分析报告。
- PlanningAgent：规划任务、分解子任务、汇总结果
- ResearchAgent：负责信息检索（使用模拟的搜索工具）
- AnalystAgent：负责数据分析和推理

SelectorGroupChat 会使用 LLM 动态选择下一个最合适的 Agent 发言，
而不是固定的轮流顺序。
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

from model_config import get_model_client, list_available_models


# ====================================================================
# 1. 定义工具函数
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
# 2. 主函数
# ====================================================================

async def main() -> None:
    # 从 config.toml 加载模型客户端
    print(f"📋 可用模型: {list_available_models()}")
    model_client = get_model_client("gpt-5-2")  # 从 config.toml 读取

    # ------------------------------------------------------------------
    # 3. 创建 Agent
    # ------------------------------------------------------------------

    # 规划 Agent：负责任务分解和结果汇总
    planning_agent = AssistantAgent(
        name="PlanningAgent",
        description="负责规划任务、分解子任务和汇总最终结果的智能体。应在收到新任务时首先参与。",
        model_client=model_client,
        system_message=(
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
    )

    # 研究 Agent：负责信息检索
    research_agent = AssistantAgent(
        name="ResearchAgent",
        description="负责搜索和检索技术信息的智能体。",
        model_client=model_client,
        tools=[search_tech_info],
        system_message=(
            "你是一个技术研究员。使用 search_tech_info 工具搜索相关信息。\n"
            "每次只执行一个搜索任务，并将搜索结果如实汇报给团队。\n"
            "不要对结果做额外的分析或推测，只负责信息检索。"
        ),
    )

    # 分析 Agent：负责分析推理
    analyst_agent = AssistantAgent(
        name="AnalystAgent",
        description="负责数据分析、对比研究和逻辑推理的智能体。",
        model_client=model_client,
        tools=[compare_models],
        system_message=(
            "你是一个技术分析师。你的职责是：\n"
            "1. 根据 ResearchAgent 收集到的信息进行深度分析\n"
            "2. 使用 compare_models 工具做对比研究\n"
            "3. 给出有洞察力的分析结论\n"
            "基于数据和事实进行分析，不要凭空推测。"
        ),
    )

    # ------------------------------------------------------------------
    # 4. 终止条件
    # ------------------------------------------------------------------
    text_termination = TextMentionTermination("TERMINATE")
    max_msg_termination = MaxMessageTermination(max_messages=20)
    termination = text_termination | max_msg_termination

    # ------------------------------------------------------------------
    # 5. 自定义 Selector 提示（指导 LLM 选择下一个发言者）
    # ------------------------------------------------------------------
    selector_prompt = """根据当前对话上下文，选择最合适的下一个发言智能体。

可用智能体及其职责：
{roles}

当前对话：
{history}

请从 {participants} 中选择一个智能体执行下一步任务。
规则：
- 新任务到来时，应先让 PlanningAgent 规划
- PlanningAgent 分配了任务后，选择被指定的执行者
- 执行者完成后，如果还有待完成的子任务，选择下一个执行者
- 所有子任务完成后，选择 PlanningAgent 汇总结果
只选择一个智能体。"""

    # ------------------------------------------------------------------
    # 6. （可选）自定义选择函数：确保每次执行后回到 Planner 检查进度
    # ------------------------------------------------------------------
    def selector_func(
        messages: Sequence[BaseAgentEvent | BaseChatMessage],
    ) -> str | None:
        """自定义发言者选择逻辑。

        规则：如果上一条消息不是 PlanningAgent 发的，
        就让 PlanningAgent 先发言（检查进度或汇总）。
        否则交给 LLM 默认选择。
        """
        if messages[-1].source != "PlanningAgent":
            return "PlanningAgent"
        return None  # 回退到 LLM 选择

    # ------------------------------------------------------------------
    # 7. 创建 SelectorGroupChat 团队
    # ------------------------------------------------------------------
    team = SelectorGroupChat(
        participants=[planning_agent, research_agent, analyst_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        selector_func=selector_func,
        allow_repeated_speaker=True,  # 允许同一 Agent 连续发言
    )

    # ------------------------------------------------------------------
    # 8. 运行任务
    # ------------------------------------------------------------------
    task = (
        "请研究以下问题并给出分析报告：\n"
        "大语言模型中的 MoE（混合专家）架构相比传统 Transformer 有哪些优势？\n"
        "请对比 GPT-4 和 DeepSeek 的技术路线差异。"
    )

    print("=" * 60)
    print("🚀 SelectorGroupChat 智能任务调度示例")
    print(f"   任务：{task[:50]}...")
    print("=" * 60)

    result = await Console(team.run_stream(task=task))

    print("\n" + "=" * 60)
    print(f"✅ 任务完成！停止原因: {result.stop_reason}")
    print(f"   共产生 {len(result.messages)} 条消息")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
