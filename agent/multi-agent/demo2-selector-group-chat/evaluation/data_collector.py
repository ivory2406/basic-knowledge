"""
数据采集模块 (Phase 1: Data Collection)

采集 Agent 运行过程中的全量交互数据，包括：
- 每条消息的内容、来源、时间戳
- 工具调用事件（请求、执行、结果）
- Selector 决策记录（选了谁、为什么）
- 任务级别的聚合指标
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any
from pathlib import Path


@dataclass
class ToolCallRecord:
    """工具调用记录"""
    agent: str
    tool_name: str
    arguments: str
    result: str
    is_error: bool
    call_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class MessageRecord:
    """消息记录"""
    source: str  # 发送者
    msg_type: str  # TextMessage / ToolCallRequestEvent / ToolCallExecutionEvent / ...
    content: str  # 消息内容摘要
    full_content: str  # 完整内容
    seq_no: int  # 消息序号
    timestamp: float = field(default_factory=time.time)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


@dataclass
class SelectorDecision:
    """Selector 选择决策记录"""
    selected_agent: str
    reason: str  # 选择原因（LLM / selector_func）
    context_summary: str  # 当前上下文摘要
    seq_no: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskTrace:
    """一次任务运行的完整 Trace"""
    task_id: str
    task_input: str
    config: dict  # 运行配置（模型、prompt 等）
    messages: list[MessageRecord] = field(default_factory=list)
    selector_decisions: list[SelectorDecision] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    stop_reason: str = ""
    total_messages: int = 0

    def finalize(self, stop_reason: str):
        self.end_time = time.time()
        self.stop_reason = stop_reason
        self.total_messages = len(self.messages)


class DataCollector:
    """数据采集器：从 AutoGen 运行结果中提取结构化 Trace"""

    def __init__(self, output_dir: str = "evaluation/traces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_run_output(self, log_text: str, task_id: str = "task_001",
                         task_input: str = "", config: dict | None = None) -> TaskTrace:
        """从运行日志文本解析出结构化 Trace"""
        trace = TaskTrace(
            task_id=task_id,
            task_input=task_input,
            config=config or {},
        )

        lines = log_text.strip().split("\n")
        current_source = ""
        current_type = ""
        current_content_lines: list[str] = []
        seq_no = 0
        pending_tool_calls: list[ToolCallRecord] = []

        def flush_message():
            nonlocal seq_no, current_source, current_type, current_content_lines, pending_tool_calls
            if current_source:
                content = "\n".join(current_content_lines).strip()
                msg = MessageRecord(
                    source=current_source,
                    msg_type=current_type,
                    content=content[:200],  # 摘要
                    full_content=content,
                    seq_no=seq_no,
                    tool_calls=pending_tool_calls.copy(),
                )
                trace.messages.append(msg)
                seq_no += 1
                current_content_lines = []
                pending_tool_calls = []

        for line in lines:
            line_stripped = line.strip()

            # 检测消息头: ---------- TypeName (Source) ----------
            if line_stripped.startswith("----------") and line_stripped.endswith("----------"):
                flush_message()
                # 解析类型和来源
                inner = line_stripped.strip("-").strip()
                if "(" in inner and ")" in inner:
                    parts = inner.rsplit("(", 1)
                    current_type = parts[0].strip()
                    current_source = parts[1].rstrip(")").strip()
                else:
                    current_type = inner
                    current_source = "system"
            elif line_stripped.startswith("✅") or line_stripped.startswith("==="):
                # 终止行
                flush_message()
                if "停止原因" in line_stripped:
                    trace.stop_reason = line_stripped.split("停止原因:")[-1].strip() if "停止原因:" in line_stripped else line_stripped
            elif line_stripped.startswith("[FunctionCall("):
                # 工具调用请求
                current_content_lines.append(line_stripped)
            elif line_stripped.startswith("[FunctionExecutionResult("):
                # 工具调用结果 — 解析并记录
                current_content_lines.append(line_stripped)
                try:
                    # 简单提取关键信息
                    tc = ToolCallRecord(
                        agent=current_source,
                        tool_name=self._extract_field(line_stripped, "name"),
                        arguments="",
                        result=self._extract_field(line_stripped, "content"),
                        is_error=("is_error=True" in line_stripped),
                        call_id=self._extract_field(line_stripped, "call_id"),
                    )
                    pending_tool_calls.append(tc)
                except Exception:
                    pass
            else:
                current_content_lines.append(line_stripped)

        flush_message()
        trace.finalize(trace.stop_reason or "unknown")
        return trace

    def _extract_field(self, text: str, field_name: str) -> str:
        """从日志文本中提取指定字段的值"""
        try:
            # 处理 name='xxx' 或 content='xxx' 格式
            key = f"{field_name}='"
            if key in text:
                start = text.index(key) + len(key)
                end = text.index("'", start)
                return text[start:end]
            # 处理 name="xxx" 格式
            key = f'{field_name}="'
            if key in text:
                start = text.index(key) + len(key)
                end = text.index('"', start)
                return text[start:end]
        except (ValueError, IndexError):
            pass
        return ""

    def save_trace(self, trace: TaskTrace) -> Path:
        """将 Trace 保存为 JSON"""
        filepath = self.output_dir / f"{trace.task_id}_trace.json"

        def serialize(obj: Any) -> Any:
            if hasattr(obj, "__dict__"):
                return asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj.__dict__
            return str(obj)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(trace), f, ensure_ascii=False, indent=2, default=serialize)

        return filepath

    def load_trace(self, filepath: str | Path) -> dict:
        """加载已保存的 Trace"""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
