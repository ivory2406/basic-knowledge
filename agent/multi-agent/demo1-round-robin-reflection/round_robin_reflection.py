"""
RoundRobinGroupChat 示例 —— 反思模式 (Reflection Pattern)

场景：写一篇关于"AI Agent 发展趋势"的短文。
- primary_agent：负责撰写内容
- critic_agent：负责审阅并给出修改意见，满意后回复 APPROVE

两个 Agent 轮流发言，直到 critic 回复 APPROVE 为止。
"""

import asyncio
import sys
from pathlib import Path

# 将上级目录加入 sys.path，以便引用共享模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

from model_config import get_model_client, list_available_models


async def main() -> None:
    # ----------------------------------------------------------------
    # 1. 从 config.toml 加载模型客户端
    #    配置文件位于 agent/config.toml，支持多模型配置
    #    可通过 model_key 切换不同模型，如 "gpt-5-2"、"deepseek" 等
    # ----------------------------------------------------------------
    print(f"📋 可用模型: {list_available_models()}")
    model_client = get_model_client("gpt-5-2")  # 从 config.toml 读取

    # ----------------------------------------------------------------
    # 2. 创建 Agent
    # ----------------------------------------------------------------

    # 写作 Agent：负责撰写和修改文章
    writer_agent = AssistantAgent(
        name="Writer",
        model_client=model_client,
        system_message=(
            "你是一位专业的技术写作专家。\n"
            "根据用户的主题撰写高质量的中文短文（300-500字）。\n"
            "当收到审稿人的反馈后，请根据反馈修改文章并输出完整的修改版。"
        ),
    )

    # 审稿 Agent：负责审阅文章，给出修改建议
    critic_agent = AssistantAgent(
        name="Critic",
        model_client=model_client,
        system_message=(
            "你是一位严格的技术文章审稿人。\n"
            "请从以下维度审阅文章：\n"
            "1. 内容准确性和深度\n"
            "2. 结构是否清晰\n"
            "3. 语言是否流畅\n"
            "4. 是否有遗漏的关键观点\n\n"
            "如果文章质量已经足够好，请回复 'APPROVE'。\n"
            "否则，请给出具体的修改建议。"
        ),
    )

    # ----------------------------------------------------------------
    # 3. 定义终止条件
    # ----------------------------------------------------------------

    # 条件 1：当 Critic 回复中包含 "APPROVE" 时停止
    text_termination = TextMentionTermination("APPROVE")

    # 条件 2：最多交互 10 轮，防止无限循环
    max_msg_termination = MaxMessageTermination(max_messages=10)

    # 两个条件取 OR —— 任一满足即停止
    termination = text_termination | max_msg_termination

    # ----------------------------------------------------------------
    # 4. 创建 RoundRobinGroupChat 团队
    #    Agent 按列表顺序轮流发言：Writer -> Critic -> Writer -> ...
    # ----------------------------------------------------------------
    team = RoundRobinGroupChat(
        participants=[writer_agent, critic_agent],
        termination_condition=termination,
    )

    # ----------------------------------------------------------------
    # 5. 运行任务（流式输出）
    # ----------------------------------------------------------------
    print("=" * 60)
    print("🚀 RoundRobinGroupChat 反思模式示例")
    print("   主题：AI Agent 的发展趋势")
    print("=" * 60)

    result = await Console(
        team.run_stream(task="请撰写一篇关于 'AI Agent 的发展趋势' 的技术短文。")
    )

    print("\n" + "=" * 60)
    print(f"✅ 任务完成！停止原因: {result.stop_reason}")
    print(f"   共产生 {len(result.messages)} 条消息")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
