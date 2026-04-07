# AutoGen Multi-Agent 示例

基于 [AutoGen 0.4+](https://microsoft.github.io/autogen/stable/) 框架构建的多智能体协作示例。

## 项目结构

```
agent/
├── config.toml                        # ⭐ 统一模型配置（API Key、地址等）
└── multi-agent/
    ├── README.md                      # 本说明文件
    ├── requirements.txt               # Python 依赖
    ├── model_config.py                # 配置加载模块（读取 config.toml）
    ├── paper_tools.py                 # 论文搜索/下载/解析工具集
    ├── round_robin_reflection.py      # 示例 1：RoundRobinGroupChat 反思模式
    ├── selector_group_chat.py         # 示例 2：SelectorGroupChat 智能调度模式
    ├── memory_paper_analysis.py       # 示例 3：Memory 论文搜集与分析
    └── papers/                        # 下载的论文 PDF 存放目录（自动创建）
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
python round_robin_reflection.py

# 示例 2：智能调度模式
python selector_group_chat.py

# 示例 3：Memory 论文搜集与分析（真实工具调用，会下载论文 PDF）
python memory_paper_analysis.py
```

---

## 示例说明

### 示例 1：RoundRobinGroupChat — 反思模式

**文件**: `round_robin_reflection.py`

**场景**: Writer + Critic 协作完成一篇技术短文。

```
任务 → Writer 撰写 → Critic 审稿 → Writer 修改 → Critic 审稿 → ... → APPROVE ✅
```

**核心概念**：
- **RoundRobinGroupChat**：Agent 按固定顺序轮流发言
- **TextMentionTermination**：检测特定关键词自动终止
- **MaxMessageTermination**：设置最大消息数防止死循环
- 终止条件可以用 `|`（OR）和 `&`（AND）组合

**Agent 角色**：

| Agent | 职责 |
|-------|------|
| Writer | 根据主题撰写文章，收到反馈后修改 |
| Critic | 从多个维度审阅文章，满意后回复 APPROVE |

---

### 示例 2：SelectorGroupChat — 智能任务调度

**文件**: `selector_group_chat.py`

**场景**: 三个 Agent 协作完成技术研究任务。

```
任务 → Planner 规划 → Researcher 搜索 → Planner 检查
     → Analyst 分析 → Planner 检查 → ... → Planner 汇总 → TERMINATE ✅
```

**核心概念**：
- **SelectorGroupChat**：使用 LLM 动态选择下一个发言的 Agent
- **Tool（工具调用）**：Agent 可以调用外部函数完成特定任务
- **selector_prompt**：自定义提示词指导 LLM 选择发言者
- **selector_func**：自定义选择函数覆盖 LLM 的默认选择逻辑

**Agent 角色**：

| Agent | 职责 | 工具 |
|-------|------|------|
| PlanningAgent | 任务规划、子任务分配、结果汇总 | 无 |
| ResearchAgent | 技术信息检索 | `search_tech_info` |
| AnalystAgent | 数据分析与对比研究 | `compare_models` |

---

### 示例 3：Memory 论文搜集与分析 — 真实学术研究场景

**文件**: `memory_paper_analysis.py` + `paper_tools.py`

**场景**: 三个 Agent 协作完成 LLM Memory 领域论文的搜集、下载、阅读和对比分析。

```
任务 → Planner 规划搜索策略
     → Collector 搜索 arXiv → Collector 下载 PDF
     → Planner 检查下载进度
     → Analyst 提取论文内容 → Analyst 对比分析
     → Planner 汇总最终报告 → TERMINATE ✅
```

**与示例 2 的区别**：
- **真实工具调用**：不是模拟数据，而是真正调用 arXiv API 搜索和下载论文
- **PDF 解析**：使用 PyMuPDF 提取论文全文，Agent 可以阅读论文内容
- **持久化存储**：下载的 PDF 保存在 `papers/` 目录，带有索引文件

**Agent 角色**：

| Agent | 职责 | 工具 |
|-------|------|------|
| Planner | 规划搜索策略、协调进度、撰写最终报告 | 无 |
| Collector | 搜索 arXiv、下载 PDF、管理论文库 | `search_papers`, `download_paper`, `list_downloaded_papers` |
| Analyst | 提取 PDF 文本、深度阅读、对比分析 | `extract_pdf_text`, `list_downloaded_papers` |

**研究目标**：
- 搜集 PersonaMem、KnowMe-Bench、LongMemEval 等 memory benchmark 论文
- 搜集 MemoryBank、M+/MemoryLLM 等 memory 技术论文
- 输出结构化对比分析报告（自动保存为 `analysis_report.md`）

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
