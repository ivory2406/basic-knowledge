# 模型开源通常开源什么

模型开源不是一个单一概念。很多厂商说“开源模型”，实际可能只是开放权重；也可能同时开放推理代码、训练框架、技术报告、评测脚本、数据配方，甚至完整训练数据。判断一个模型到底开放到什么程度，要拆开看。

---

## 1. 模型开源的常见层次

| 层次 | 开源内容 | 能做什么 | 是否等于完整复现训练 |
|---|---|---|---|
| Open API | 只提供接口调用 | 可以使用模型，但不能本地运行 | 否 |
| Open Weights | 开放模型权重文件 | 可以下载、本地部署、微调或量化 | 否 |
| Open Inference Code | 开放推理代码和加载方式 | 可以知道如何跑模型、如何适配推理框架 | 否 |
| Open Model Architecture | 公开模型结构、配置、tokenizer、chat template | 可以理解模型如何组织参数和输入输出 | 否 |
| Open Training Recipe | 公开训练方法、超参、优化器、数据规模、后训练流程 | 可以接近复现训练思路 | 不一定 |
| Open Training Framework | 开放训练代码、并行策略、分布式实现 | 可以按类似方式训练或继续训练 | 不一定 |
| Open Data / Data Recipe | 开放训练数据或数据构造方法 | 可以更接近原始训练过程 | 接近，但仍受算力和细节影响 |
| Open Evaluation | 开放 benchmark、评测脚本、prompt 和结果 | 可以复核模型能力 | 否 |

所以，“模型开源”最常见的实际含义是 **open weights**，不是完整训练工程开源。

---

## 2. 常见开源内容说明

### 2.1 模型权重

模型权重是模型训练完成后的参数文件，通常以 `safetensors`、`bin`、`gguf` 等格式发布。

开放权重意味着：

- 可以本地加载模型。
- 可以私有化部署。
- 可以做推理服务。
- 可以做 LoRA / SFT 等二次微调。
- 可以做量化、蒸馏或适配不同硬件。

但开放权重不意味着：

- 公开原始训练数据。
- 公开完整训练代码。
- 能以相同成本复现一个同等模型。
- 允许任何商业用途，仍需看许可证。

### 2.2 模型结构与配置

这部分通常包括：

- `config.json`
- tokenizer 文件
- chat template
- generation config
- MoE / attention / rope 等结构参数
- 上下文长度、激活参数量、总参数量

没有这些配置，即使有权重也不容易正确加载。

### 2.3 推理代码

推理代码负责把权重跑起来，包括：

- 模型加载逻辑
- attention / MoE / KV Cache 实现
- tokenizer 适配
- chat 模板拼接
- batch 推理
- OpenAI-compatible API server
- vLLM、SGLang、Transformers、llama.cpp 等框架适配说明

很多大模型项目会开放推理代码，但不开放训练代码。

### 2.4 训练框架

训练框架通常包括：

- 分布式训练代码
- 数据并行、张量并行、流水线并行、专家并行
- optimizer 实现
- mixed precision 训练
- checkpoint 保存和恢复
- fault tolerance
- 训练调度、监控和日志
- SFT / RLHF / RLVR / GRPO 等后训练流程

如果只开放权重，没有开放训练框架，就不能说“完整训练工程开源”。

### 2.5 训练数据

训练数据是最少被完整开放的部分。原因包括版权、隐私、安全、商业竞争和数据清洗成本。

常见开放方式：

- 只公布 token 数量和数据类型比例。
- 公布数据清洗策略。
- 公布部分公开数据集列表。
- 开放合成数据或后训练数据。
- 完整开放训练语料，这种比较少见。

### 2.6 技术报告

技术报告一般说明：

- 模型架构
- 参数规模
- 训练 token 数
- 训练阶段
- 关键优化方法
- benchmark 结果
- 推理和部署方式

技术报告很重要，但它不是训练代码，也不是训练数据。

### 2.7 许可证

许可证决定你能不能商用、能不能改、能不能再分发。

常见类型：

- MIT / Apache-2.0：相对宽松，通常允许商用。
- 自定义 Model License：可能限制用途、规模、再分发或高风险场景。
- Research-only：只允许研究使用。
- Non-commercial：不允许商业用途。

判断模型能不能用于公司项目，必须看模型权重许可证，而不是只看 GitHub 代码许可证。

---

## 3. “开源权重”和“开源训练框架”的区别

### 3.1 开源权重

开放的是训练完成后的结果：

```text
训练数据 + 训练代码 + 算力 + 调参
        ↓
      模型权重
```

拿到权重后，可以部署和二次开发，但通常无法知道完整训练细节。

### 3.2 开源训练框架

开放的是训练模型的工程系统：

```text
数据加载 -> 分布式训练 -> 优化器 -> checkpoint -> 后训练 -> 评测
```

训练框架开源后，开发者可以学习或复用训练方法，但如果没有原始数据、完整超参和足够算力，仍然难以复现同等模型。

### 3.3 一个模型可以只开源其中一部分

例如：

- 只开放 API：不能算真正 open weights。
- 开放权重和 tokenizer：可以本地跑，但训练不可复现。
- 开放权重、推理代码和技术报告：可以部署、研究架构，但训练工程仍不完整。
- 开放权重、训练代码、数据配方和评测：开放程度更高。
- 开放完整数据、代码、权重、日志和 checkpoint：最接近完整开源，但大模型中很少见。

---

## 4. DeepSeek-V4 最近开源了什么

> 说明：截至 2026-05-07，DeepSeek 官方文档显示 DeepSeek-V4 Preview 于 2026-04-24 发布并 open-sourced。用户常写作 “deepseak-v4”，官方名称是 **DeepSeek-V4**。

### 4.1 官方发布内容

DeepSeek 官方 API News 写明：

- DeepSeek-V4 Preview 已上线并开源。
- 包含两个模型：
  - **DeepSeek-V4-Pro**
  - **DeepSeek-V4-Flash**
- 两者都支持 **1M context length**。
- API 可用，模型名包括 `deepseek-v4-pro` 和 `deepseek-v4-flash`。
- 官方给出了技术报告链接和 Hugging Face open weights collection。

参考：

- DeepSeek API Docs: https://api-docs.deepseek.com/news/news260424
- Hugging Face Collection: https://huggingface.co/collections/deepseek-ai/deepseek-v4

### 4.2 Hugging Face 上公开的模型

DeepSeek-V4 collection 中包含四类模型：

| 模型 | 类型 | 总参数 | 激活参数 | 上下文长度 | 精度 |
|---|---|---:|---:|---:|---|
| DeepSeek-V4-Flash-Base | Base | 284B | 13B | 1M | FP8 Mixed |
| DeepSeek-V4-Flash | Instruct / Chat | 284B | 13B | 1M | FP4 + FP8 Mixed |
| DeepSeek-V4-Pro-Base | Base | 1.6T | 49B | 1M | FP8 Mixed |
| DeepSeek-V4-Pro | Instruct / Chat | 1.6T | 49B | 1M | FP4 + FP8 Mixed |

其中：

- **Base** 更接近预训练底座模型，适合研究、继续训练和评测基础能力。
- **Instruct / Chat** 是经过后训练的对话模型，适合直接作为助手使用。
- **Pro** 更大、更强，成本更高。
- **Flash** 更小、更快，偏工程性价比。

### 4.3 DeepSeek-V4 开放的主要内容

从官方页面和模型卡看，DeepSeek-V4 主要开放了：

1. **模型权重**
   - Hugging Face 和 ModelScope 提供下载。
   - 包含 Base 和 Chat / Instruct 版本。

2. **模型配置**
   - 模型结构配置、tokenizer、chat template、精度信息等。
   - 这些是本地加载和推理所必需的。

3. **技术报告**
   - 说明模型架构、训练规模、后训练流程、评测结果。
   - V4 报告标题为 `DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence`。

4. **推理使用方式**
   - Hugging Face 模型卡给出了 Transformers、vLLM、SGLang、Docker Model Runner 等使用入口。
   - 这属于推理生态适配，不等于官方完整训练框架开源。

5. **许可证**
   - Hugging Face 模型卡显示 license 为 **MIT**。
   - 实际使用仍应以对应模型仓库的 license 文件为准。

### 4.4 DeepSeek-V4 技术报告披露的训练信息

模型卡介绍了这些训练和架构信息：

- 架构是 MoE。
- DeepSeek-V4-Pro：1.6T 总参数，49B 激活参数。
- DeepSeek-V4-Flash：284B 总参数，13B 激活参数。
- 支持 1M token 上下文。
- 采用 Hybrid Attention，包括 CSA 和 HCA，用于提升长上下文效率。
- 引入 Manifold-Constrained Hyper-Connections（mHC）增强训练稳定性和表达能力。
- 使用 Muon optimizer。
- 预训练数据规模超过 32T tokens。
- 后训练包括领域专家独立培养，以及通过 on-policy distillation 做统一模型整合。
- Chat 模型支持 Non-think、Think、Think Max 等不同推理努力模式。

这些是 **训练方法披露**，不是完整训练代码或原始训练数据开放。

### 4.5 DeepSeek-V4 没有明确完整开放的内容

从目前公开页面看，不能把 DeepSeek-V4 的开源理解为“完整训练工程全部开源”。没有明确完整开放的内容包括：

- 原始预训练数据全集。
- 完整数据清洗和配比流水线。
- 完整训练框架源码。
- 全量分布式训练脚本。
- 完整超参数、训练日志和中间 checkpoint。
- 后训练标注数据、偏好数据或 RL 环境全集。
- 训练所用基础设施和调度系统。

换句话说，DeepSeek-V4 更准确的描述是：

```text
开放权重 + 开放模型配置 + 开放技术报告 + 开放推理使用方式
```

而不是：

```text
完整训练数据 + 完整训练框架 + 完整训练过程全部开源
```

---

## 5. 如何判断一个模型到底开源了什么

可以按下面清单检查：

| 检查项 | 关键问题 |
|---|---|
| 权重 | 是否能下载完整 checkpoint？是否包含 base 和 instruct？ |
| 许可证 | 是否允许商用、修改、再分发？有没有用途限制？ |
| tokenizer | 是否开放 tokenizer 文件和 chat template？ |
| 架构 | 是否公开模型结构和关键模块？ |
| 推理 | 是否提供加载代码、推理示例、推理框架适配？ |
| 训练代码 | 是否提供 pretraining / SFT / RL 代码？ |
| 数据 | 是否开放训练数据或至少开放数据配方？ |
| 评测 | 是否开放评测脚本、prompt、benchmark 细节？ |
| 版本 | 是否有模型 revision、commit、checksum？ |
| 依赖 | 是否说明硬件、精度、并行策略和运行要求？ |

---

## 6. 工程落地建议

### 6.1 如果目标是本地部署

重点看：

- 权重是否可下载。
- 推理框架是否支持。
- 显存和多机要求。
- 量化版本是否可靠。
- license 是否允许你的使用场景。

DeepSeek-V4-Pro 是 1.6T MoE 模型，本地部署门槛极高；Flash 版本相对更适合工程落地，但仍然是超大模型。

### 6.2 如果目标是微调

重点看：

- 是否有 Base 模型。
- 是否支持 LoRA / QLoRA / full fine-tune。
- tokenizer 和 chat template 是否稳定。
- 训练框架是否支持该架构。
- MoE、FP4/FP8、长上下文是否会给训练带来额外复杂度。

### 6.3 如果目标是学习训练方法

重点看：

- 技术报告。
- 模型结构配置。
- 开源社区的训练复现项目。
- SGLang、Megatron、DeepSpeed、verl、OpenRLHF 等生态支持。

但要注意：学习训练方法不等于能完整复现模型。大模型训练还依赖数据、算力、工程细节和大量调参经验。

---

## 7. 一句话总结

模型“开源”要分层理解。最常见的是开放权重，而不是开放完整训练框架。以 DeepSeek-V4 为例，官方公开的是 V4-Pro / V4-Flash 的权重、模型配置、技术报告、推理使用方式和许可证信息；训练方法有披露，但目前不能理解为原始训练数据和完整训练框架已经全部开源。

