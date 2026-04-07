# Demo 3: Memory 论文搜集与分析 — 真实学术研究场景

## 场景

三个 Agent 协作完成 LLM Memory 领域论文的搜集、下载、阅读和对比分析。

```
任务 → Planner 规划搜索策略
     → Collector 搜索 arXiv → Collector 下载 PDF
     → Planner 检查下载进度
     → Analyst 提取论文内容 → Analyst 对比分析
     → Planner 汇总最终报告 → TERMINATE ✅
```

## 与 Demo 2 的区别

- **真实工具调用**：不是模拟数据，而是真正调用 arXiv API 搜索和下载论文
- **PDF 解析**：使用 PyMuPDF 提取论文全文，Agent 可以阅读论文内容
- **持久化存储**：下载的 PDF 保存在 `../papers/` 目录，带有索引文件

## Agent 角色

| Agent | 职责 | 工具 |
|-------|------|------|
| Planner | 规划搜索策略、协调进度、撰写最终报告 | 无 |
| Collector | 搜索 arXiv、下载 PDF、管理论文库 | `search_papers`, `download_paper`, `list_downloaded_papers` |
| Analyst | 提取 PDF 文本、深度阅读、对比分析 | `extract_pdf_text`, `list_downloaded_papers` |

## 研究目标

- 搜集 PersonaMem、KnowMe-Bench、LongMemEval 等 memory benchmark 论文
- 搜集 MemoryBank、M+/MemoryLLM 等 memory 技术论文
- 输出结构化对比分析报告（自动保存为 `analysis_report.md`）

## 运行

```bash
cd agent/multi-agent/demo3-memory-paper-analysis
python memory_paper_analysis.py
```

## 运行结果

见 [analysis_report.md](./analysis_report.md)，包含 5 篇论文的结构化对比分析报告。
