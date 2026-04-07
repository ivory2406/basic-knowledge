# AutoGen Multi-Agent 示例

基于 [AutoGen 0.4+](https://microsoft.github.io/autogen/stable/) 框架构建的多智能体协作示例，覆盖 **协作模式探索 → 真实任务执行 → 闭环评测优化** 三个递进阶段。

## 项目结构

```
agent/
├── config.toml                                # ⭐ 统一模型配置（API Key、地址等）
└── multi-agent/
    ├── README.md                              # 本说明文件
    ├── requirements.txt                       # Python 依赖
    ├── model_config.py                        # 配置加载模块（读取 config.toml）
    │
    ├── demo1-round-robin-reflection/          # 示例 1：RoundRobinGroupChat 反思模式
    │   ├── README.md
    │   └── round_robin_reflection.py
    │
    ├── demo2-selector-group-chat/             # 示例 2：SelectorGroupChat + 闭环评测体系
    │   ├── README.md
    │   ├── selector_group_chat.py             #   原始智能调度 Demo
    │   ├── configurable_agent.py              #   可配置 Agent 运行器
    │   ├── run_evaluation_loop.py             #   🔄 闭环评测主驱动
    │   └── evaluation/                        #   评测体系模块
    │       ├── data_collector.py              #     Phase 1: 数据采集
    │       ├── evaluator.py                   #     Phase 2/4: 多维度评测
    │       ├── optimizer.py                   #     Phase 3: 自动优化
    │       ├── report_generator.py            #     报告生成
    │       ├── traces/                        #     采集的 Trace 数据
    │       ├── results/                       #     评测结果 JSON
    │       ├── plans/                         #     优化方案 JSON
    │       ├── reports/                       #     📊 评测报告 & 对比报告
    │       └── logs/                          #     运行日志
    │
    └── demo3-memory-paper-analysis/           # 示例 3：Memory 论文搜集与分析
        ├── README.md
        ├── memory_paper_analysis.py
        ├── paper_tools.py                     #   arXiv API + PDF 解析工具
        ├── analysis_report.md                 #   ✅ 对比分析报告
        └── papers/                            #   下载的 7 篇论文 PDF + 索引
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型

编辑 `agent/config.toml`，配置模型的 API Key 和地址：

```toml
[models.gpt-5-2]
api_key = "your-api-key"
base_url = "https://vdbteam.openai.azure.com/openai/v1/"
model_name = "gpt-5.2"

# 可添加更多模型
[models.deepseek]
api_key = "your-deepseek-key"
base_url = "https://api.deepseek.com"
model_name = "deepseek-chat"
```

代码中通过 `model_key` 一行切换模型：

```python
from model_config import get_model_client

model_client = get_model_client("gpt-5-2")    # 使用 GPT-5.2
# model_client = get_model_client("deepseek") # 切换为 DeepSeek
```

### 3. 运行示例

```bash
cd agent/multi-agent

# 示例 1：反思模式
python demo1-round-robin-reflection/round_robin_reflection.py

# 示例 2：智能调度模式
python demo2-selector-group-chat/selector_group_chat.py

# 示例 2（闭环评测）：采集 → 评测 → 优化 → 再评测
python demo2-selector-group-chat/run_evaluation_loop.py

# 示例 3：Memory 论文搜集与分析（真实工具调用，会下载论文 PDF）
python demo3-memory-paper-analysis/memory_paper_analysis.py
```

---

## 示例总览

三个 Demo 构成递进式学习路径：

```
Demo 1 (基础)           Demo 2 (进阶)               Demo 3 (实战)
固定轮次协作       →    LLM 智能调度 + 闭环评测  →    真实学术研究任务
RoundRobin              SelectorGroupChat              arXiv API + PDF 解析
Writer↔Critic           Planner↔Researcher↔Analyst     Planner↔Collector↔Analyst
```

---

### 示例 1：[RoundRobinGroupChat — 反思模式](./demo1-round-robin-reflection/)

Writer + Critic 协作完成一篇技术短文。Agent 按固定顺序轮流发言，直到 Critic 回复 APPROVE。

```
任务 → Writer 撰写 → Critic 审稿 → Writer 修改 → ... → APPROVE ✅
```

| Agent | 职责 |
|-------|------|
| Writer | 根据主题撰写文章，收到反馈后修改 |
| Critic | 从多个维度审阅文章，满意后回复 APPROVE |

**核心概念**：RoundRobinGroupChat、TextMentionTermination、MaxMessageTermination

---

### 示例 2：[SelectorGroupChat — 智能调度 + 闭环评测](./demo2-selector-group-chat/)

三个 Agent 协作完成技术研究任务。SelectorGroupChat 使用 LLM 动态选择下一个发言者。
在此基础上构建了完整的 **Agent 闭环评测体系**。

```
任务 → Planner 规划 → Researcher 搜索 → Analyst 分析 → Planner 汇总 → TERMINATE ✅
```

| Agent | 职责 | 工具 |
|-------|------|------|
| PlanningAgent | 任务规划、子任务分配、结果汇总 | 无 |
| ResearchAgent | 技术信息检索 | `search_tech_info` |
| AnalystAgent | 数据分析与对比研究 | `compare_models` |

**核心概念**：SelectorGroupChat、Tool（工具调用）、selector_prompt、selector_func

#### 🔄 闭环评测体系

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Agent 运行 │──▶│ 数据采集  │──▶│ 多维评测  │──▶│ 自动优化  │──▶│ 再次评测  │
│ Baseline  │   │ Phase 1  │   │ Phase 2  │   │ Phase 3  │   │ Phase 4  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └─────┬────┘
      ▲                                                            │
      └──────────────────── 反馈闭环 ──────────────────────────────┘
```

评测体系详见 [demo2 README](./demo2-selector-group-chat/README.md)，包含：

| 模块 | 文件 | 功能 |
|------|------|------|
| 数据采集 | `data_collector.py` | 从运行日志提取结构化 Trace（消息记录、工具调用、Selector 决策） |
| 多维评测 | `evaluator.py` | 五维度加权评分（任务完成度 30% / 效率 20% / 协作 20% / 工具 15% / 输出 15%） |
| 自动优化 | `optimizer.py` | 基于评测结果自动生成 Prompt 优化、调度策略改进方案 |
| 报告生成 | `report_generator.py` | Markdown 格式的单轮评测报告 & 多轮对比报告 |

**评测结果摘要**：

| 指标 | 基线 | 优化后 | 变化 |
|------|------|--------|------|
| 综合得分 | 75.2% 🟡 | 71.0% 🟡 | 📉 -4.2% |
| 效率指标 | 87.3% | 100.0% | 📈 +12.7% |
| 输出质量 | 100.0% | 67.5% | 📉 -32.5% |

> **关键发现**：Prompt 优化导致 LLM"过于高效"——直接在规划阶段就输出完整报告，跳过多 Agent 协作流程。说明 Agent 优化需要同时约束行为边界，防止流程被短路。

---

### 示例 3：[Memory 论文搜集与分析](./demo3-memory-paper-analysis/)

三个 Agent 协作完成 LLM Memory 领域论文的搜集、下载、阅读和对比分析。**真实工具调用**。

```
任务 → Planner 规划 → Collector 搜索/下载 → Analyst 分析 → Planner 汇总报告 → TERMINATE ✅
```

| Agent | 职责 | 工具 |
|-------|------|------|
| Planner | 规划搜索策略、协调进度、撰写最终报告 | 无 |
| Collector | 搜索 arXiv、下载 PDF、管理论文库 | `search_papers`, `download_paper`, `list_downloaded_papers` |
| Analyst | 提取 PDF 文本、深度阅读、对比分析 | `extract_pdf_text`, `list_downloaded_papers` |

**成果**：自动下载并分析了 7 篇 Memory 相关论文（PersonaMem、KnowMe-Bench、LongMemEval、M+、Mem-Gallery、MemoryBank 等），生成结构化对比分析报告 → [analysis_report.md](./demo3-memory-paper-analysis/analysis_report.md)

**核心概念**：真实 arXiv API 调用、PDF 解析、持久化存储

---

## AutoGen 核心架构

```
┌─────────────────────────────────────────────┐
│                AutoGen 0.4+                 │
├──────────────┬──────────────────────────────┤
│  AgentChat   │  高级 API（推荐入门使用）       │
│  (本项目使用)  │  - AssistantAgent            │
│              │  - RoundRobinGroupChat       │
│              │  - SelectorGroupChat         │
│              │  - Swarm / GraphFlow         │
├──────────────┼──────────────────────────────┤
│  Core        │  底层事件驱动框架（高级用户）    │
├──────────────┼──────────────────────────────┤
│  Extensions  │  模型客户端、工具扩展等         │
│              │  - OpenAIChatCompletionClient │
└──────────────┴──────────────────────────────┘
```

## 关键 API 参考

| 类/函数 | 用途 |
|---------|------|
| `AssistantAgent` | 基于 LLM 的智能体，支持系统提示和工具调用 |
| `RoundRobinGroupChat` | 固定顺序轮流发言的团队 |
| `SelectorGroupChat` | LLM 动态选择发言者的团队 |
| `TextMentionTermination` | 检测特定文本触发终止 |
| `MaxMessageTermination` | 达到最大消息数终止 |
| `Console` | 流式输出到控制台的 UI 工具 |
| `OpenAIChatCompletionClient` | OpenAI 兼容的模型客户端 |

## 参考链接

- [AutoGen 官方文档](https://microsoft.github.io/autogen/stable/)
- [AgentChat 教程](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html)
- [Teams 教程](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/teams.html)
- [SelectorGroupChat 教程](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/selector-group-chat.html)
