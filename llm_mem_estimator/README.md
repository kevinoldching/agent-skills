# LLM GPU Memory Estimator

LLM 显存估算工具 - 计算大语言模型的 GPU 显存占用

## 功能特性

- 估算模型权重、KV Cache、激活值的显存占用
- 支持多种并行策略：TP / PP / DP / EP / CP
- 支持多种注意力机制：MHA / MQA / GQA / SWA / MLA / DSA / GDN
- 支持 MoE (Mixture of Experts) 架构
- 从 HuggingFace 及本地权重路径自动生成配置文件
- 估算最大支持的序列长度和批大小

## 安装 Claude Code Skill

```bash
# 安装到 ~/.claude/skills/llm_mem_estimator
llm-mem-estimator -g

# 卸载 Claude Code Skill
llm-mem-estimator -r
```

> 注意：全局安装 `npm install -g` 不会自动安装 Claude Code Skill，需要手动运行上述命令。

安装后，Claude Code 会自动识别此 Skill，当你提到显存估算时会自动调用。

> **注意**：`npx github:kevinoldching/agent-skills/llm_mem_estimator` 无法直接运行，因为 package.json 在子目录中。请使用下方的 npm 安装方式。

## 安装方式

### 方式一：本地克隆安装

```bash
# 克隆项目
git clone https://github.com/kevinoldching/agent-skills.git
cd agent-skills/llm_mem_estimator

# 项目级安装
npm install

# 或全局安装
npm install -g
```

### 方式二：项目级安装（通过 GitHub URL）

```bash
npm install github:kevinoldching/agent-skills/llm_mem_estimator
```

### 方式三：全局安装（通过 GitHub URL）

```bash
# 全局安装
npm install -g github:kevinoldching/agent-skills/llm_mem_estimator

# 运行
llm-mem-estimator --help
```

## 使用示例

### 1. 从 HuggingFace 模型生成配置

```bash
llm-mem-estimator --model Qwen/Qwen2.5-0.5B --generate-config
```

### 2. 估算显存占用

```bash
# 使用生成的配置
llm-mem-estimator --config configs/models/Qwen2.5-0.5B.yaml --seq-len 4096 --batch-size 1

# 指定并行策略
llm-mem-estimator --config configs/models/gpt-oss-120b.yaml --tp 8
llm-mem-estimator --config configs/models/gpt-oss-120b.yaml --tp 2 --pp 2

# 指定硬件芯片
llm-mem-estimator --config configs/models/my_model.yaml --chip nvidia/H100-80GB

# 查找最大序列长度
llm-mem-estimator --config configs/models/my_model.yaml --chip nvidia/H100-80GB --batch-size 4 --find-max-seq-len
```

### 3. 完整参数示例

```bash
llm-mem-estimator \
  --config configs/models/DeepSeek-V3.yaml \
  --seq-len 8192 \
  --batch-size 8 \
  --gen-len 2048 \
  --tp 8 \
  --pp 2 \
  --dp 2 \
  --chip nvidia/H100-80GB
```

## 命令行参数

| 参数 | 说明 |
|------|------|
| `--config` | 模型 YAML 配置文件路径 |
| `--model` | HuggingFace 模型名称 |
| `--local` | 本地模型权重路径 |
| `--generate-config` | 从模型生成 YAML 配置 |
| `--output-config` | 生成的配置文件输出路径 |
| `--batch-size` | 批大小（默认: 1） |
| `--seq-len` | 序列长度 |
| `--prompt-len` | 输入提示长度 |
| `--gen-len` | 生成输出长度 |
| `--kv-dtype` | KV Cache 数据类型（默认: fp16） |
| `--activation-dtype` | 激活值数据类型（默认: fp16） |
| `--tp` | Tensor Parallel 并行数（默认: 1） |
| `--pp` | Pipeline Parallel 并行数（默认: 1） |
| `--dp` | Data Parallel 并行数（默认: 1） |
| `--cp` | Context Parallel 并行数（默认: 1） |
| `--ep` | Expert Parallel 并行数（默认: 1） |
| `--chip` | 硬件芯片名称（如 nvidia/H100-80GB） |
| `--find-max-seq-len` | 查找最大支持的序列长度 |

## 卸载

```bash
# 项目级卸载
npm uninstall llm-mem-estimator

# 全局卸载
npm uninstall -g llm-mem-estimator
```

## 手动安装 Python 依赖

如果自动安装失败，可以手动安装：

```bash
pip install -r requirements.txt
```

## 依赖项

- Python 3.8+
- pyyaml >= 6.0
- huggingface_hub >= 0.20.0
- tqdm >= 4.65.0

## 相关文档

- [CLI 详细说明](./docs/spec/calculate_mem_cli_spec.md)
- [YAML 配置规范](./docs/spec/yaml_config_spec.md)
- [权重映射规则](./docs/spec/weight_mapping_rules_spec.md)

## License

MIT
