# Demo 1: RoundRobinGroupChat — 反思模式 (Reflection Pattern)

## 场景

Writer + Critic 协作完成一篇技术短文。

```
任务 → Writer 撰写 → Critic 审稿 → Writer 修改 → Critic 审稿 → ... → APPROVE ✅
```

## 核心概念

- **RoundRobinGroupChat**：Agent 按固定顺序轮流发言
- **TextMentionTermination**：检测特定关键词自动终止
- **MaxMessageTermination**：设置最大消息数防止死循环
- 终止条件可以用 `|`（OR）和 `&`（AND）组合

## Agent 角色

| Agent | 职责 |
|-------|------|
| Writer | 根据主题撰写文章，收到反馈后修改 |
| Critic | 从多个维度审阅文章，满意后回复 APPROVE |

## 运行

```bash
cd agent/multi-agent/demo1-round-robin-reflection
python round_robin_reflection.py
```

## 运行结果

见 [run_output.log](./run_output.log)，经历了 1 轮反思修改后 Critic 回复 APPROVE 通过。
