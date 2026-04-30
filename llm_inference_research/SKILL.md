---
name: llm_inference_research
description: 大模型推理技术研究技能。当用户需要搜集和分析LLM推理优化技术、推理框架(vLLM/SGLang/TensorRT-LLM等)、模型架构创新(MoE/Mamba/Linear Attention等)、性能优化(量化/压缩/算子融合)相关论文和开源动态时触发。支持两种输出模式：1)深度技术报告（单篇论文/技术详细分析）；2)简报（定期汇总多源信息）。当用户查询包含"最近X天/周/月"、"简报"、"汇总"、"动态"等时间范围词时，触发简报模式，从paper_sources.md中的多源信息并行搜集并整合输出。输出包含技术摘要、核心原理和落地建议的结构化报告或简报。
---

# 大模型推理技术研究技能

本技能用于系统性地搜集、分析和整合大模型推理领域的技术动态，重点关注论文研究、开源项目进展和工程实践。

## 研究范围

### 核心技术方向

1. **推理框架与引擎**
   - **vLLM**: PagedAttention、Prefix Caching、Chunked Prefill、 speculative decoding
   - **SGLang**: RadixAttention、Multi-turn Conversation Caching、结构化生成优化
   - **TensorRT-LLM**: Kernel融合优化、多GPU并行、量化推理
   - **DeepSpeed-Inference**: ZeRO partitioning、MoE并行推理、混合精度训练
   - **llama.cpp**: 端侧推理优化、多种量化方案、移动设备支持
   - **ollama**: 本地部署、模型管理、API兼容
   - **LM Studio**: 桌面端推理、模型下载管理、OpenAI兼容API

2. **模型架构创新**
   - **MoE (Mixture of Experts)**: 稀疏激活、Top-K路由专家选择、负载均衡损失
   - **Mamba/State Space Models**: 选择性状态空间、线性时间复杂度、选择性扫描机制
   - **Multi-Head Latent Attention (MLA)**: DeepSeek-V2/V3架构、KV Cache压缩
   - **Linear Attention**: 低秩近似、降低KV Cache开销
   - **Ring Attention**: 超长序列处理、分布式注意力计算
   - **Sliding Window Attention**: 长序列局部注意力优化
   - **Speculative Decoding**: 草稿模型预测、并行解码、接受/拒绝机制

3. **性能优化技术**
   - **量化**: GPTQ/AWQ/FP8/INT8/INT4/INT1、KV Cache量化、激活值量化
   - **压缩**: 结构化剪枝、知识蒸馏、LoRA适配器压缩
   - **算子融合**: FlashAttention/FlashAttention-2/FlashAttention-3、FlashInfer、Kernel融合
   - **内存优化**: PagedAttention、vLLM内存池、KV Cache管理、Prefix Caching
   - **调度优化**: Continuous Batching、Chunked Prefill、Dynamic Batching
   - **并行策略**: Tensor Parallelism、Pipeline Parallelism、Expert Parallelism、ZeRO优化
   - **推理加速**: CUDA Graphs、Torch Compile、Triton Kernel优化

4. **推理系统架构**
   - **在线推理服务**: RESTful API、gRPC、WebSocket、流式输出
   - **离线批处理**: 大规模推理、结果缓存、失败重试
   - **多模态推理**: 文本、图像、音频、视频联合推理
   - **Agent推理**: 工具调用、链式思考、多轮对话、记忆管理
   - **边缘部署**: 模型压缩、硬件适配、低延迟优化
   - **多租户隔离**: 资源配额、优先级调度、请求限流

5. **前沿研究方向**
   - **长上下文处理**: 100K+ token处理、位置编码优化、上下文压缩
   - **高效微调**: LoRA/QLoRA/DoRA、Adapter Tuning、Prefix Tuning
   - **安全对齐**: RLHF、DPO、Constitutional AI、对抗攻击防御
   - **可解释性**: 注意力可视化、神经元激活分析、因果推理
   - **多语言优化**: 低资源语言支持、跨语言迁移学习
   - **硬件协同设计**: 专用AI芯片、存内计算、光计算

### 信息来源

1. **学术论文**
   - arXiv: cs.AI、cs.LG、cs.CL、cs.DC、cs.AR
   - OpenReview: ICLR、NeurIPS、ICML
   - ScienceDirect、IEEE Xplore、ACM DL

2. **开源项目**
   - GitHub: vllm-project/vllm、sgl-project/sglang、deepseek-ai
   - Hugging Face: 模型发布、论文实现

3. **技术博客与文档**
   - 项目官方博客: vLLM blog、SGLang documentation
   - 公司技术博客: NVIDIA、DeepSeek
   - 社区分享: 知乎、CSDN、个人技术博客

4. **社区动态**
   - Discord/Slack: vLLM events、SGLang开发者社区
   - Mailing List: 相关项目邮件列表
   - X/Twitter: 研究者分享、项目发布

## 工作模式

本skill支持两种工作模式，根据用户查询自动选择：

### 模式1：深度技术报告（默认）

**适用场景**：分析单篇论文、特定技术、开源项目

**用户查询特征**：
- "分析一下《论文标题》"
- "调研一下PagedAttention技术"

**输出**：详细的技术分析报告（5000-10000字）
- 技术摘要
- 核心原理详解
- 关键创新点分析
- 落地建议

### 模式2：定期简报

**适用场景**：获取某时间段内的技术动态汇总

**用户查询特征**：
- "获取最近1个月的大模型推理信息"
- "本周vLLM社区有什么新动态"
- "季度技术简报"

**触发关键词**：最近X天/周/月、本周、本月、季度、简报、汇总、动态、进展

**工作流程**：

```
用户查询:"获取最近1个月内的大模型推理技术动态"
         ↓
    识别时间范围: 最近1个月
         ↓
    读取paper_sources.md获取信息源
         ↓
    并行搜集:
    ├─ 论文搜索 (arXiv过去1个月, 关键词列表)
    ├─ 技术博客 (官方博客RSS/订阅源)
    └─ 社区讨论 (Discord/Slack关键频道)
         ↓
    信息整合与分类:
    ├─ 按技术方向分类 (推理框架/架构创新/性能优化)
    ├─ 按重要程度标注 (P0关键/P1重要/P2参考)
    └─ 按时间线排序
         ↓
    生成简报:
    ├─ 执行摘要 (关键发现、核心建议)
    ├─ 技术进展总览 (分类表格)
    ├─ 重点详情 (Top 5-10关键进展)
    └─ 行动建议 (下一步关注方向)
         ↓
    输出: 结构化简报 (2000-3000字)
```

**输出格式**（简报模板）：

```markdown
# 大模型推理技术简报 - 2024年X月第X周

## 执行摘要
- **关键发现**：本周/本月X个重要技术进展
- **核心建议**：建议立即关注X，观望Y

## 技术进展总览

| 技术方向 | 本周关键进展 | 重要程度 |
|---------|------------|---------|
| 推理框架 | vLLM v0.4.2发布 | P0 |
| 架构创新 | DeepSeek-V3开源 | P0 |
| 性能优化 | FlashInfer v0.2 | P1 |

## 重点详情

### 1. [标题] (P0)
**来源**：GitHub Release / arXiv / 官方博客
**核心内容**：
- 要点1
- 要点2
**行动建议**：建议跟进

### 2. [标题] (P1)
...

## 行动建议

### 立即关注 (1-2周)
1. XXXXXX
2. XXXXXX

### 中期跟进 (1-3月)
1. XXXXXX
2. XXXXXX

### 持续观察
- XXXXXX

## 参考资料
- [论文链接]
- [GitHub Release]
- [技术博客]
```

**简报特点**：
1. **信息密度高**：精简呈现关键信息，避免冗长
2. **结构化强**：表格+列表+摘要，快速定位信息
3. **可操作性强**：明确的行动建议和时间规划
4. **时间敏感**：明确的时间范围标注

## 使用示例

### 场景1: 分析单篇论文

用户: "分析一下这篇论文《SGLang: Efficient Execution of Structured Language Model Programs》"

执行流程:
1. 搜索论文PDF或arXiv版本
2. 提取论文核心内容
3. 按照分析框架生成报告

### 场景2: 跟踪技术动态（简报模式）

用户: "获取最近1个月的大模型推理技术简报"

执行流程:
1. 识别时间范围：最近1个月
2. 读取paper_sources.md获取信息源配置
3. 并行搜集：
   - 论文搜索（arXiv过去1个月）
   - 技术博客（官方博客）
4. 信息整合与分类（按技术方向、重要程度）

### 场景3: 调研特定技术方向

用户: "调研一下Speculative Decoding在推理优化中的最新进展"

执行流程:
1. 搜索相关论文（arXiv、会议论文）
2. 查找开源实现（vLLM、SGLang等支持情况）
3. 分析不同方案的原理和效果
4. 生成综合分析报告

## 参考资料使用指南

本skill包含以下参考资料，在分析过程中按需读取：

### 1. `references/paper_sources.md`
**何时读取**：需要搜索论文或技术资料时
**包含内容**：
- 顶会列表（OSDI/SOSP/MLSys/NeurIPS等）
- 按技术方向分类的关键词库
- 关键研究团队追踪
- GitHub项目监控清单
- 论文搜索最佳实践

**使用方式**：
```
用户查询"MoE相关技术" → 读取paper_sources.md → 获取关键词["Mixture of Experts", "MoE routing", ...] → 执行搜索
```

## 分析框架

### 1. 技术摘要

对技术/论文进行高层次概括：
- 解决的核心问题
- 主要技术路线
- 适用范围和场景
- 与现有方案的对比优势

### 2. 核心原理

深入解释技术机制：
- 算法原理和数学基础
- 系统架构设计
- 关键数据结构和流程
- 性能瓶颈分析

### 3. 关键创新点

识别和总结创新贡献：
- 与SOTA的差异化
- 核心技术创新点
- 工程实现亮点
- 理论突破或实践验证

### 4. 落地建议

提供可操作的实施方案：
- 技术可行性评级（高/中/低）
- 短期可尝试的PoC方案
- 中长期优化方向
- 风险点和应对策略
- 资源投入建议

## 输出格式

分析报告应使用以下结构化的Markdown格式：

```markdown
# [技术/论文标题]

## 技术摘要
- **核心问题**: [一句话描述]
- **技术路线**: [简要说明方法]
- **适用场景**: [适用范围]
- **主要优势**: [与现有方案对比]

## 核心原理
[详细技术解释，包括算法、架构、流程等]

### 关键公式/算法
```
[如果有数学公式或伪代码]
```

## 关键创新点
1. [创新点1]
2. [创新点2]
3. [创新点3]

## 落地建议
- **可行性评级**: [高/中/低]
- **短期PoC**: [可快速验证的方案]
- **中长期方向**: [深度优化路径]
- **风险提示**: [技术风险和应对]
- **资源投入**: [人力/时间/硬件建议]

## 参考资料
- 论文链接: [URL]
- 开源实现: [GitHub URL]
- 相关文档: [Doc URL]
```

## 注意事项

1. **论文获取优先级**:
   - 优先获取arXiv版本（开放获取）
   - 对于会议论文，尝试查找作者个人主页或机构库
   - 记录无法获取的论文，在报告中标注

2. **信息验证**:
   - 交叉验证不同来源的信息
   - 区分论文声称的性能与实际工程落地效果
   - 注意区分学术原型与生产级实现

3. **报告质量**:
   - 技术内容准确，原理解释清晰
   - 既有宏观视角，又有细节分析
   - 落地建议具体可操作
   - 保持客观中立，避免过度宣传
