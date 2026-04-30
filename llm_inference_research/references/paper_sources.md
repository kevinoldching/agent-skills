# 大模型推理论文搜索源

## 顶级会议与期刊

### 系统/体系结构顶会
- **OSDI**: USENIX Symposium on Operating Systems Design and Implementation
- **SOSP**: ACM Symposium on Operating Systems Principles
- **ASPLOS**: Architectural Support for Programming Languages and Operating Systems
- **ATC**: USENIX Annual Technical Conference
- **EuroSys**: European Conference on Computer Systems

### 机器学习系统
- **MLSys**: Conference on Machine Learning and Systems
- **NeurIPS**: Neural Information Processing Systems (Systems Track)
- **ICML**: International Conference on Machine Learning (Systems Track)
- **ICLR**: International Conference on Learning Representations

### 数据库/大数据 (LLM Serving相关)
- **SIGMOD**: ACM International Conference on Management of Data
- **VLDB**: International Conference on Very Large Data Bases
- **CIDR**: Conference on Innovative Data Systems Research

## 关键研究团队与作者

### 推理系统优化
- **vLLM Team**: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang (UC Berkeley)
- **SGLang Team**: Lianmin Zheng, Ying Sheng (UC Berkeley, LMSYS)
- **FlashAttention**: Tri Dao, Dan Fu (Stanford)
- **DeepSpeed**: Microsoft Research (Ammar Ahmad Awan, et al.)

### 模型架构创新
- **MoE**: William Fedus, Barret Zoph (Google), Albert Q. Jiang (Mistral)
- **Mamba**: Tri Dao, Albert Gu (CMU, Princeton)
- **Linear Attention**: Katharopoulos et al., Choromanski et al.
- **DeepSeek**: DeepSeek-AI团队 (多头潜在注意力MLA)

## 论文搜索关键词

### 推理优化核心
```
LLM inference optimization
Large language model serving
Efficient transformer inference
KV cache optimization
Attention optimization
Memory-efficient attention
PagedAttention
Continuous batching
Speculative decoding
Draft-then-verify
```

### 量化与压缩
```
LLM quantization
Post-training quantization
GPTQ, AWQ, SmoothQuant
KV cache quantization
Weight clustering
Model compression
Knowledge distillation LLM
```

### 架构创新
```
Mixture of Experts MoE
Efficient MoE routing
Mamba state space model
Linear attention
RWKV
RetNet
Multi-head latent attention MLA
Sliding window attention
Group query attention GQA
Multi-query attention MQA
```

### 并行与分布式
```
Tensor parallelism
Pipeline parallelism
Sequence parallelism
Expert parallelism MoE
ZeRO optimization
GPipe, PipeDream
All-to-all communication
Collective communication optimization
```

## 开源项目追踪

### 核心推理框架
| 项目 | 地址 | 关注重点 |
|------|------|---------|
| vLLM | vllm-project/vllm | releases, RFCs, events, performance benchmarks |
| SGLang | sgl-project/sglang | structured generation, RadixAttention |
| TensorRT-LLM | NVIDIA/TensorRT-LLM | kernel optimization, multi-GPU |
| llama.cpp | ggerganov/llama.cpp | quantization, edge deployment |

### 关键依赖库
| 项目 | 地址 | 关注重点 |
|------|------|---------|
| FlashAttention | Dao-AILab/flash-attention | 新版本 |
| FlashInfer | flashinfer-ai/flashinfer | 新kernel库、PagedAttention优化 |
| xFormers | facebookresearch/xformers | 内存高效attention |

## 技术博客与资源

### 官方博客
- vLLM Blog: https://vllm.ai/blog
- Hugging Face Blog: https://huggingface.co/blog
- NVIDIA Developer Blog: https://developer.nvidia.com/blog/

### 研究者博客
- Lil'Log (Google): https://lilianweng.github.io/
- Sebastian Raschka: https://magazine.sebastianraschka.com/
- Tim Dettmers (量化): https://timdettmers.com/
- 苏剑林 (科学空间): https://spaces.ac.cn/

### 社区资源
- vLLM events: https://vllm.ai/events
- Papers With Code (LLM): https://paperswithcode.com/area/large-language-models
- Awesome LLM: https://github.com/Hannibal046/Awesome-LLM
- LLM Inference Survey: https://github.com/DefTruth/LLM-Inference-Survey
