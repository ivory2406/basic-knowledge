"""
Memory 论文搜集与分析 Multi-Agent 系统

使用 AutoGen SelectorGroupChat 构建三个协作 Agent：
- PlannerAgent:  任务规划与结果汇总（指挥官）
- CollectorAgent: 论文搜索与下载（搜集员）
- AnalystAgent:  论文阅读与对比分析（分析师）

目标：搜集 PersonaMem、KnowMe-Bench、LongMemEval 等 memory benchmark
和 MemoryBank、MemGPT、M+ 等 memory 技术论文，进行对比分析。
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
from paper_tools import (
    search_papers,
    download_paper,
    extract_pdf_text,
    list_downloaded_papers,
)


async def main() -> None:
    # ==================================================================
    # 1. 加载模型
    # ==================================================================
    print(f"📋 可用模型: {list_available_models()}")
    model_client = get_model_client("gpt-5-2")

    # ==================================================================
    # 2. 创建 Agent
    # ==================================================================

    # ---- Planner：任务规划 + 结果汇总 ----
    planner = AssistantAgent(
        name="Planner",
        description="负责规划研究任务、分解子任务、协调团队和汇总最终分析报告的智能体。新任务到来时应首先参与。",
        model_client=model_client,
        system_message="""\
你是一个学术研究项目的负责人/规划者。你的职责：

1. **任务规划**：将用户的研究需求拆分为具体的、可执行的子任务
2. **任务分配**：将子任务分配给合适的团队成员
3. **进度管控**：检查每个子任务是否完成，未完成则继续推进
4. **结果汇总**：所有子任务完成后，撰写结构化的对比分析报告

你的团队成员：
  - Collector：论文搜集专家，可以搜索 arXiv、下载 PDF、查看已下载论文列表
  - Analyst：论文分析专家，可以提取和阅读 PDF 内容，进行深度分析

工作流程：
  Phase 1 - 搜索：让 Collector 按关键词搜索相关论文
  Phase 2 - 下载：让 Collector 下载筛选出的重点论文
  Phase 3 - 分析：让 Analyst 逐篇提取和阅读论文内容
  Phase 4 - 对比：让 Analyst 进行跨论文对比分析
  Phase 5 - 汇总：你自己整合所有分析结果，输出最终报告

分配任务格式：
  → Collector: <具体任务>
  → Analyst: <具体任务>

注意事项：
- 你只负责规划和汇总，不要自己执行搜索、下载或分析
- 每次只分配 1-2 个子任务，等完成后再分配下一批
- 最终报告必须包含：论文概览表、关键对比维度、各方案优劣势、结论
- 当最终报告撰写完成后，在末尾加上 'TERMINATE'
""",
    )

    # ---- Collector：论文搜集 ----
    collector = AssistantAgent(
        name="Collector",
        description="负责在 arXiv 搜索论文、下载 PDF 和管理论文库的智能体。",
        model_client=model_client,
        tools=[search_papers, download_paper, list_downloaded_papers],
        system_message="""\
你是一个学术论文搜集专家。你的职责：

1. 使用 search_papers 工具在 arXiv 上搜索相关论文
2. 使用 download_paper 工具下载指定论文的 PDF
3. 使用 list_downloaded_papers 查看已下载的论文列表

工作原则：
- 根据 Planner 的指示执行搜索和下载任务
- 搜索时使用精准的英文关键词，尝试多种关键词组合以提高召回率
- 下载论文时使用 arxiv_id（如 "2504.14225"）
- 完成任务后如实汇报搜索/下载结果
- 如果搜索结果不理想，主动建议替代关键词

注意：你只负责搜集论文，不要对论文内容进行分析。
""",
    )

    # ---- Analyst：论文分析 ----
    analyst = AssistantAgent(
        name="Analyst",
        description="负责提取论文内容、深度阅读、对比分析和生成分析结论的智能体。",
        model_client=model_client,
        tools=[extract_pdf_text, list_downloaded_papers],
        system_message="""\
你是一个资深的学术论文分析师，专注于 LLM 记忆系统和 benchmark 领域。你的职责：

1. 使用 extract_pdf_text 工具提取已下载论文的文本内容
2. 深度阅读并理解论文的核心贡献
3. 进行跨论文的对比分析

分析维度（每篇论文都要覆盖）：
- **研究目标**：论文要解决什么问题
- **核心方法/框架**：提出了什么技术方案或评测框架
- **数据集特点**：使用了什么数据、规模、来源
- **评估指标**：使用了哪些评估指标
- **关键发现**：最重要的实验结果和结论
- **与其他工作的关系**：与同类工作的异同

对比分析时重点关注：
- Benchmark 的评测维度差异（事实记忆 vs 偏好追踪 vs 时间推理 vs 深层理解）
- 数据构造方式差异（模拟对话 vs 真实对话 vs 自传叙事）
- Memory 技术路线差异（检索增强 vs 参数化记忆 vs 混合方案）
- 各方案在不同评估维度上的优劣势

输出格式要求：
- 使用 Markdown 格式
- 包含结构化的对比表格
- 关键观点要有论文证据支撑
""",
    )

    # ==================================================================
    # 3. 终止条件
    # ==================================================================
    text_termination = TextMentionTermination("TERMINATE")
    max_msg_termination = MaxMessageTermination(max_messages=40)
    termination = text_termination | max_msg_termination

    # ==================================================================
    # 4. Selector 配置
    # ==================================================================
    selector_prompt = """根据当前对话上下文，选择最合适的下一个发言智能体。

可用智能体及其职责：
{roles}

当前对话：
{history}

请从 {participants} 中选择一个智能体。
规则：
- 新任务或需要规划时 → Planner
- 需要搜索或下载论文时 → Collector
- 需要提取/阅读论文内容或进行分析时 → Analyst
- 执行者完成子任务后 → Planner（检查进度）
- 所有分析完成后 → Planner（撰写最终报告）
只选择一个智能体。"""

    def selector_func(
        messages: Sequence[BaseAgentEvent | BaseChatMessage],
    ) -> str | None:
        """确保每个执行者完成后都回到 Planner 检查进度。"""
        if messages[-1].source != "Planner":
            return "Planner"
        return None  # 回退到 LLM 选择

    # ==================================================================
    # 5. 创建团队
    # ==================================================================
    team = SelectorGroupChat(
        participants=[planner, collector, analyst],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        selector_func=selector_func,
        allow_repeated_speaker=True,
    )

    # ==================================================================
    # 6. 定义研究任务
    # ==================================================================
    task = """\
请帮我完成以下学术研究任务：

**主题：LLM 长期记忆（Long-term Memory）领域的 Benchmark 与技术方案对比分析**

需要搜集和分析的论文/方向包括（但不限于）：

1. **Memory Benchmark（评测基准）**：
   - PersonaMem（COLM 2025）：动态用户画像与个性化响应评测
   - KnowMe-Bench（2026）：基于自传叙事的人物理解评测
   - LongMemEval（ICLR 2025）：聊天助手长期交互记忆评测

2. **Memory 技术方案**：
   - MemoryBank（AAAI 2024）：基于记忆存储+遗忘机制的长期记忆
   - M+ / MemoryLLM（2025）：可扩展的长期记忆架构
   - 以及搜索过程中发现的其他相关工作

请完成以下任务：
1. 搜索并下载上述论文的 PDF
2. 提取各论文的核心内容
3. 输出一份结构化的对比分析报告，重点包括：
   - 各 Benchmark 在 memory 评测维度上的关键区别
   - 各 Memory 技术方案的核心思路和优劣势
   - 对该领域发展趋势的总结
"""

    # ==================================================================
    # 7. 运行
    # ==================================================================
    print("=" * 70)
    print("🧠 Memory 论文搜集与分析 Multi-Agent 系统")
    print("=" * 70)
    print(f"📝 任务: {task[:100]}...")
    print("=" * 70)

    result = await Console(team.run_stream(task=task))

    print("\n" + "=" * 70)
    print(f"✅ 任务完成！停止原因: {result.stop_reason}")
    print(f"   共产生 {len(result.messages)} 条消息")
    print("=" * 70)

    # 保存最终报告
    report_path = Path(__file__).parent / "analysis_report.md"
    final_messages = [
        msg for msg in result.messages
        if hasattr(msg, "source") and msg.source == "Planner"
    ]
    if final_messages:
        last_planner_msg = final_messages[-1]
        content = last_planner_msg.content if hasattr(last_planner_msg, "content") else str(last_planner_msg)
        # 移除 TERMINATE 标记
        content = content.replace("TERMINATE", "").strip()
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Memory 论文对比分析报告\n\n")
            f.write(f"> 由 Multi-Agent 系统自动生成\n\n")
            f.write(content)
        print(f"\n📄 最终报告已保存至: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
