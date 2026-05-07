# 模型缓存的具体实现方式

模型缓存（Model Cache）不是单一技术，而是一组围绕「减少重复计算、减少重复调用、提升吞吐、降低延迟和成本」的工程手段。不同缓存解决的是不同层次的问题：有的缓存模型推理过程中的中间状态，有的缓存完整回答，有的缓存检索和 embedding 结果。

---

## 1. 缓存分层

| 层级 | 缓存对象 | 典型场景 | 命中收益 | 主要风险 |
|---|---|---|---|---|
| KV Cache | Transformer attention 的 Key/Value 张量 | 自回归生成、多轮对话、长上下文推理 | 减少历史 token 重算 | 显存占用大、会话隔离要求高 |
| Prefix / Prompt Cache | 公共前缀 prompt 的预计算 KV | 系统提示词、工具说明、长文档前缀复用 | 降低首 token 延迟 | 前缀必须完全一致或可规范化 |
| Response Cache | 完整输入对应的完整输出 | FAQ、固定问答、低变化业务 | 可直接跳过模型调用 | 过期、个性化、权限泄漏 |
| Semantic Cache | 语义相近问题对应的答案 | 客服、知识库问答、意图稳定场景 | 提高相似问题复用率 | 相似不等于等价，容易答非所问 |
| Embedding Cache | 文本到向量的结果 | RAG 建库、重复查询、批量文档处理 | 减少 embedding API 或模型开销 | 模型版本变更后必须重建 |
| Retrieval Cache | Query 的检索结果 | RAG、多轮追问、热点查询 | 减少向量库/搜索开销 | 数据更新后结果陈旧 |
| Model Artifact Cache | 模型权重、Tokenizer、编译图 | 服务启动、弹性扩缩容、边缘部署 | 缩短冷启动 | 版本管理和磁盘占用 |

---

## 2. KV Cache

### 2.1 核心原理

Transformer 自回归生成时，每生成一个新 token，都需要对已有上下文做 attention。如果每一步都重新计算所有历史 token 的 Key 和 Value，复杂度会非常高。

KV Cache 的做法是：

1. Prefill 阶段：对输入 prompt 做一次前向计算，得到每一层 attention 的 Key/Value。
2. Decode 阶段：每生成一个新 token，只计算新 token 的 Key/Value，并和历史 KV 拼接后参与 attention。
3. 后续 token 复用历史 KV，不再重复计算历史上下文。

简化流程：

```text
输入 prompt: [t1, t2, t3]
Prefill: 计算并缓存 K/V(t1), K/V(t2), K/V(t3)

生成 t4:
  只计算 K/V(t4)
  attention 使用 K/V(t1..t4)

生成 t5:
  只计算 K/V(t5)
  attention 使用 K/V(t1..t5)
```

### 2.2 注意力如何影响下一个 token

注意力可以理解成：模型在预测当前 token 时，会判断前文哪些 token 更值得参考。匹配度越高，注意力权重越大，对当前预测的影响也越大。

例如上下文是：

```text
小明把牛奶放进冰箱，因为它需要冷藏。小明打开
```

模型要预测下一个 token，候选可能是：

```text
冰箱 / 书包 / 电视 / 门
```

当前生成位置相当于在问：

```text
我接下来该接什么？前文哪些词最有用？
```

模型会给前文 token 分配注意力权重，示意如下：

| 前文 token | 相关性 | 注意力权重 |
|---|---:|---:|
| 小明 | 中 | 0.15 |
| 牛奶 | 高 | 0.25 |
| 放进 | 中 | 0.15 |
| 冰箱 | 很高 | 0.35 |
| 冷藏 | 高 | 0.10 |

这些权重会影响模型内部形成的“当前语境表示”。因为“冰箱”“牛奶”“冷藏”这些信息被更多吸收，所以模型会认为：

```text
小明打开冰箱
```

比下面这些更合理：

```text
小明打开书包
小明打开电视
小明打开门
```

最终，注意力不会直接替模型选出某个词，而是经过多层神经网络计算后，输出整个词表上的概率分布：

| 候选 token | 概率 |
|---|---:|
| 冰箱 | 0.62 |
| 门 | 0.18 |
| 书包 | 0.06 |
| 电视 | 0.03 |

如果使用贪心解码，模型会选择概率最高的“冰箱”；如果使用采样，模型大概率选择“冰箱”，但也可能选择其他 token。

这也是 KV Cache 有用的原因：前文 token 的 Key/Value 已经参与过注意力计算，后续生成时可以复用这些 Key/Value，不必每一步都从头计算前文。

进一步看，这段上下文可以先粗略拆成：

```text
小明 / 把 / 牛奶 / 放进 / 冰箱 / 因为 / 它 / 需要 / 冷藏 / 。 / 小明 / 打开
```

模型读完这段上下文后，会给每个 token 都算出一份 Key 和 Value：

```text
Key   = 这个 token 提供什么线索，方便以后被匹配
Value = 这个 token 真正携带的信息，方便以后被取用
```

KV Cache 里缓存的大致含义可以这样类比：

| token | 缓存的 Key 像什么 | 缓存的 Value 像什么 |
|---|---|---|
| 小明 | 人物 / 主语 / 动作执行者 | 小明这个人，以及他可能执行动作 |
| 牛奶 | 物品 / 需要冷藏 / 被放置物 | 牛奶这个物体，和它需要冷藏的属性 |
| 放进 | 放置动作 / 位置关系 | 有一个东西被放到某个地方 |
| 冰箱 | 地点 / 容器 / 冷藏相关 | 冰箱这个容器，和冷藏功能 |
| 它 | 代词 / 指代前文物品 | 可能指向牛奶 |
| 需要 | 需求 / 原因关系 | 后面解释为什么 |
| 冷藏 | 低温保存 / 原因线索 | 强化牛奶和冰箱的关系 |
| 打开 | 动作 / 需要一个可打开对象 | 当前动作还缺少宾语 |

真实的 Key/Value 不是文字标签，而是一串高维数字向量。上表只是为了帮助理解。

当模型要预测：

```text
小明打开 __
```

当前位置会产生新的 Query，像是在问：

```text
打开的对象最可能是什么？
```

这个 Query 会去匹配缓存里的 Key：

```text
Query: 打开的对象是什么？
  匹配 冰箱 的 Key：很高
  匹配 牛奶 的 Key：中等
  匹配 小明 的 Key：较低
  匹配 冷藏 的 Key：中等
```

然后模型用注意力权重去加权对应的 Value：

```text
0.45 * Value(冰箱)
+ 0.20 * Value(牛奶)
+ 0.15 * Value(冷藏)
+ ...
```

得到一个“当前上下文表示”，大意是：

```text
这里很可能是在说打开冰箱
```

所以，KV Cache 可以理解成：把每个历史 token 提前整理成“可被查询的索引 Key”和“可被取用的信息 Value”，等生成下一个 token 时直接拿来查，不用重新整理前文。

### 2.3 数据结构

KV Cache 通常按如下维度组织：

```text
[num_layers][batch_size][num_heads][seq_len][head_dim]
```

在实际推理服务中，为了支持不同请求长度、批处理和动态调度，常见实现会进一步拆成块：

```text
Block 0: tokens 0-15
Block 1: tokens 16-31
Block 2: tokens 32-47
...
```

这种块化管理方便复用、回收和分页，也能降低显存碎片。

### 2.4 关键实现点

- **会话级缓存**：每个请求或对话维护自己的 KV Cache，避免不同用户之间上下文串扰。
- **增量解码**：decode 时只追加新 token 的 KV，不重算旧 token。
- **Batch 调度**：多个请求可以共享一次 decode step，但每个请求的 KV 长度不同，需要调度器维护位置映射。
- **显存管理**：KV Cache 通常比模型权重更容易随并发和上下文长度膨胀，需要限制最大上下文、最大并发和 cache block 数量。
- **释放策略**：请求完成、连接断开、超时或达到最大生成长度时释放对应 KV。

### 2.5 常见优化

| 优化方式 | 说明 |
|---|---|
| PagedAttention | 把 KV Cache 拆成固定大小 block，像操作系统分页一样管理，降低碎片并提升并发能力 |
| Sliding Window Cache | 只保留最近窗口内 token 的 KV，适合支持滑动窗口注意力的模型 |
| KV Quantization | 用 FP8/INT8 等方式压缩 KV，降低显存占用 |
| CPU Offload | 把冷 KV 或长上下文 KV 放到 CPU 内存，需要时再搬回 GPU |
| Prefix Sharing | 多个请求共享相同前缀的 KV block，减少重复 prefill |

---

## 3. Prefix / Prompt Cache

### 3.1 适用场景

很多业务请求有大量相同前缀，例如：

- 固定 system prompt
- 工具调用说明
- 安全策略说明
- 长文档问答中的文档内容
- Agent 框架生成的固定角色设定

如果每个请求都重新 prefill 这些前缀，成本很高。Prefix Cache 会把公共前缀对应的 KV 缓存起来，后续请求只需要从分叉点继续计算。

### 3.2 实现方式

#### 方式一：精确前缀匹配

将 tokenized prompt 的前 N 个 token 作为 key：

```text
cache_key = hash(model_id + tokenizer_id + token_ids[0:N])
```

命中后复用该前缀的 KV Cache。

优点是安全、确定性强；缺点是对 prompt 的格式很敏感，空格、换行、工具顺序变化都可能导致 miss。

#### 方式二：树形前缀缓存

把 prompt token 构造成前缀树：

```text
root
 └── system prompt
      └── tool definitions
           ├── user query A
           └── user query B
```

多个请求共享树上的公共路径，只为分叉后的 token 计算新 KV。这种方式适合多租户、多工具、多 Agent 的高并发场景。

#### 方式三：块级前缀缓存

结合 PagedAttention，把 token 序列拆成固定大小 block。完全相同的 block 可以共享；最后一个不完整 block 单独处理。

优点是实现上更接近推理引擎内部的 KV 管理方式，适合服务端高吞吐推理。

### 3.3 Cache Key 设计

Prefix Cache 的 key 不能只看文本内容，还要包含影响推理结果的上下文：

```text
cache_key = hash(
  model_id,
  model_revision,
  tokenizer_revision,
  adapter_id,
  decoding_affecting_config,
  token_ids
)
```

如果使用 LoRA、不同 tokenizer、不同模型版本或不同特殊 token 模板，必须视为不同缓存。

---

## 4. Response Cache

### 4.1 核心思路

Response Cache 缓存的是「输入 -> 输出」的完整结果，命中时可以直接返回，不再调用模型。

常见 key：

```text
cache_key = hash(
  normalized_prompt,
  model_id,
  temperature,
  top_p,
  tools_version,
  knowledge_base_version,
  user_scope
)
```

### 4.2 适用条件

Response Cache 适合：

- FAQ 和标准答案
- 低温度或 temperature=0 的确定性输出
- 结构化抽取任务
- 分类、路由、标签生成
- 翻译、摘要等输入完全一致的任务

不适合：

- 高度个性化回答
- 强依赖实时数据的问题
- 高 temperature 创作
- 涉及用户权限的数据查询

### 4.3 实现步骤

1. 对输入做规范化：去掉无意义空白、统一换行、排序 JSON 字段。
2. 构造带版本信息的 cache key。
3. 先查缓存，命中则直接返回。
4. 未命中时调用模型。
5. 对返回结果做安全检查后写入缓存。
6. 设置 TTL 或根据业务数据版本主动失效。

伪代码：

```python
def ask_model(request):
    key = build_cache_key(request)
    cached = cache.get(key)
    if cached:
        return cached

    response = model.generate(request.prompt)
    if can_cache(request, response):
        cache.set(key, response, ttl=3600)
    return response
```

---

## 5. Semantic Cache

### 5.1 核心思路

Semantic Cache 不要求输入文本完全一致，而是把用户问题转成 embedding，在向量空间中查找相似历史问题。如果相似度足够高，就复用历史答案。

流程：

```text
用户问题 -> embedding -> 向量检索 -> 相似度判断 -> 命中则返回历史答案
                                      -> 未命中则调用模型并写入缓存
```

### 5.2 命中判断

不能只用一个相似度阈值。更稳妥的判断应包含：

- embedding cosine similarity 大于阈值
- query 意图一致
- 用户权限一致
- 知识库版本一致
- 时间敏感性可接受
- 答案类型一致，例如都是定义类、步骤类或事实类

示例策略：

```text
if similarity > 0.92 and same_intent and same_scope:
    return cached_answer
else:
    call_model()
```

### 5.3 防误命中设计

| 风险 | 例子 | 防护方式 |
|---|---|---|
| 数字差异 | “退款 7 天” vs “退款 30 天” | 对数字、日期、实体做精确校验 |
| 权限差异 | A 用户查订单，B 用户问相似问题 | cache key 加 user_id / tenant_id / acl_hash |
| 时效差异 | “今天汇率是多少” | 标记实时问题，不进入长期缓存 |
| 意图差异 | “怎么取消订单” vs “取消订单会怎样” | 加意图分类或二次判别 |

---

## 6. Embedding Cache

### 6.1 缓存对象

Embedding Cache 缓存文本向量化结果：

```text
text + embedding_model + model_version -> vector
```

适合 RAG 系统中的两类场景：

- 文档入库时：同一文档或 chunk 重复处理。
- 查询时：用户重复问题、推荐问题、自动补全问题。

### 6.2 Key 设计

```text
embedding_key = hash(
  normalized_text,
  embedding_model_id,
  embedding_model_revision,
  chunking_strategy_version
)
```

如果 chunk 规则、embedding 模型或文本清洗策略变化，旧 embedding 不能继续混用。

### 6.3 存储选择

| 存储 | 适用场景 |
|---|---|
| 本地磁盘 | 离线批处理、小规模实验 |
| Redis | 在线查询缓存、短 TTL 热点数据 |
| 对象存储 | 大批量离线 embedding 产物 |
| 向量数据库 | 既存向量又做相似度检索 |

---

## 7. Retrieval Cache

RAG 系统中，除了模型调用很贵，检索链路也可能很贵，例如 query rewrite、混合检索、rerank、多路召回。

Retrieval Cache 缓存的是：

```text
query -> top_k documents / chunks / rerank result
```

建议缓存的中间结果包括：

- query rewrite 结果
- embedding query vector
- 向量检索 top_k
- BM25 检索 top_k
- reranker 排序后的文档列表
- 最终拼接进 prompt 的 context

失效条件通常包括：

- 知识库文档新增、删除、更新
- chunk 策略变更
- embedding 模型变更
- reranker 模型变更
- 用户权限变化

---

## 8. Model Artifact Cache

模型服务启动时，常见冷启动耗时来自：

- 下载模型权重
- 加载 tokenizer
- 加载 LoRA / adapter
- 编译 CUDA graph、TensorRT engine 或其他推理优化图
- 初始化 tokenizer / chat template / special tokens

Model Artifact Cache 会把这些产物放在本地磁盘、镜像层、共享存储或节点缓存中。

实现建议：

- 权重按 `model_id + revision + quantization` 分目录。
- tokenizer 按版本锁定，不要默认拉 latest。
- 编译产物按硬件型号、驱动版本、batch shape、seq len 等维度区分。
- 启动时先查本地缓存，未命中再从远端拉取。
- 多副本部署时使用预热任务，避免线上请求触发冷启动。

---

## 9. 缓存失效策略

缓存失效是模型缓存中最容易出事故的部分。常见策略如下：

| 策略 | 说明 | 适用对象 |
|---|---|---|
| TTL | 过一段时间自动过期 | Response Cache、Semantic Cache、Retrieval Cache |
| Version Key | key 中加入模型、知识库、prompt、工具版本 | 几乎所有缓存 |
| Event Invalidation | 数据更新时主动删除相关缓存 | RAG、业务问答 |
| Scope Isolation | 按用户、租户、权限隔离 | 个性化和权限相关缓存 |
| LRU / LFU | 按最近使用或使用频率淘汰 | 显存 KV、Redis 热点缓存 |
| Size Limit | 限制单条缓存和总缓存大小 | KV、Response、Embedding |

实践中通常组合使用：

```text
cache_key = hash(model_version + prompt_version + data_version + user_scope + input)
cache_policy = TTL + LRU + event invalidation
```

---

## 10. 安全与隔离

模型缓存涉及用户输入、模型输出和可能的敏感上下文，需要特别注意：

- **用户隔离**：私有数据相关缓存必须按 user_id、tenant_id 或权限 hash 隔离。
- **权限绑定**：缓存命中时仍要校验当前用户是否有权限查看缓存内容。
- **敏感信息过滤**：含身份证、手机号、密钥、订单信息等内容的响应谨慎缓存。
- **Prompt Injection 防护**：不要把不可信输入生成的结果无条件写入共享缓存。
- **审计追踪**：记录缓存命中来源、版本和失效原因，方便排查错误答案。

---

## 11. 观测指标

上线后需要观察的不只是命中率，还包括命中后的质量。

| 指标 | 含义 |
|---|---|
| cache_hit_rate | 缓存命中率 |
| token_saved | 节省的输入/输出 token |
| latency_saved | 命中缓存节省的延迟 |
| stale_hit_rate | 过期或错误命中比例 |
| false_positive_hit | 语义缓存误命中 |
| eviction_rate | 缓存淘汰频率 |
| gpu_kv_memory_usage | KV Cache 显存占用 |
| prefix_reuse_rate | 前缀复用比例 |
| cache_write_rate | 缓存写入频率 |

对于 Semantic Cache 和 Response Cache，建议抽样做人工或模型评估，确认命中答案是否仍然正确。

---

## 12. 典型组合方案

### 12.1 在线聊天模型服务

```text
Model Artifact Cache
  -> 加速模型启动

Prefix Cache
  -> 复用 system prompt / tool definitions

KV Cache
  -> 加速同一会话的连续生成

Response Cache
  -> 复用确定性、标准化请求
```

### 12.2 RAG 知识库问答

```text
Embedding Cache
  -> 避免重复向量化

Retrieval Cache
  -> 缓存 query rewrite、top_k、rerank 结果

Semantic Cache
  -> 相似问题复用答案

Response Cache
  -> 完全相同问题直接返回
```

### 12.3 Agent 系统

```text
Prefix Cache
  -> 复用角色设定、工具说明、工作流模板

Tool Result Cache
  -> 缓存幂等工具调用结果

Memory Retrieval Cache
  -> 缓存长期记忆检索结果

KV Cache
  -> 支持长上下文推理和多轮规划
```

---

## 13. 落地建议

1. 先做 **Response Cache / Embedding Cache**，实现简单、收益清晰。
2. RAG 场景加入 **Retrieval Cache**，并把知识库版本写入 key。
3. 高并发推理服务重点优化 **KV Cache + Prefix Cache**。
4. 对语义缓存保持保守，先用于 FAQ、客服意图明确的问题。
5. 所有缓存 key 都显式加入模型版本、prompt 版本、数据版本和权限范围。
6. 缓存命中后仍要经过必要的安全检查和权限检查。
7. 持续观测误命中，而不是只追求命中率。

---

## 14. 简单决策表

| 目标 | 优先考虑 |
|---|---|
| 降低同一会话生成延迟 | KV Cache |
| 降低固定长 prompt 的首 token 延迟 | Prefix / Prompt Cache |
| 降低重复问题成本 | Response Cache |
| 复用相似问题答案 | Semantic Cache |
| 降低 RAG 入库成本 | Embedding Cache |
| 降低 RAG 查询成本 | Retrieval Cache |
| 降低服务冷启动 | Model Artifact Cache |
