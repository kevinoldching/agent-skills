---
name: code-review-batch
description: omni-npu 项目PR(Pull Request)的代码评审指南。涵盖架构一致性、设计模式、代码风格以及 NPU适配的特定注意事项。用于在评审PR时确保代码质量符合项目标准。
allowed-tools: Read
---

# omni-npu 代码评审检查清单

## 概述

对当前的 Git 变更执行结构化评审，重点关注 SOLID 设计原则、系统架构、可移除候选代码等。除非用户明确要求实施修改，否则默认仅输出评审结论。详细步骤如下：

1. **获取PR/MR内容**：通过Gitee MCP接口获取PR/MR的修改内容，结合项目的上下文做出分析
2. **生成评审草稿**：
    - 注意评审意见尽量简洁，只需要列出明确有问题的检查项


## 严重级别（Severity Levels）

| 级别 | 名称 | 描述 | 处理方式 |
|-------|------|-------------|--------|
| **P0** | 严重 | 存在安全漏洞、数据丢失风险或功能正确性缺陷	 | 必须阻止合并 |
| **P1** | 高 | 逻辑错误、严重违反 SOLID 原则、明显性能回退 | 应在合并前修复 |
| **P2** | 中 | 代码异味、可维护性问题、轻度违反 SOLID 原则 | 本次 PR 修复或创建后续任务 |
| **P3** | 低 | 风格问题、命名优化、小幅改进建议 | 建议优化 |

## 检视流程

### 1) Preflight context

- 优先使用gitee/gitcode mcp工具来获取变更范围，如不可用，则使用 `git status -sb`、`git diff --stat` 和 `git diff` 来确定变更范围。
- 如有必要，使用 `rg` 或 `grep` 查找相关模块、调用位置以及接口约定。
- 识别系统入口点、职责边界以及关键路径（如npu_worker、npu_model_runnder、model、layer等）。

**边界情况（Edge cases）：**
- **无变更**：如果 `git diff` 为空，应告知用户，并询问是否需要评审已暂存（staged）的变更或某个特定的提交范围。
- **大规模变更（>500 行）**：先按文件进行概要总结，然后按模块或功能域分批次进行评审。
- **关注点混杂**：应根据逻辑功能进行问题归类，而不仅仅按照文件顺序组织评审结论。

### 2） SOLID原则

- 加载 `references/solid_checklist.md`，获取具体的检查提示
- 重点关注以下方面:
  - **SRP（单一职责原则）**: 模块是否承担了彼此无关的多个职责，职责是否过载。
  - **OCP（开闭原则）**: 是否通过频繁修改现有代码来扩展行为，而不是通过扩展点实现功能扩展。
  - **LSP（里氏替换原则）**: 子类是否破坏父类的既有语义预期，或需要通过类型判断才能正常使用。
  - **ISP（接口隔离原则）**: 接口是否过于臃肿，包含调用方并不使用的方法。
  - **DIP（依赖倒置原则）**: 高层逻辑是否直接依赖于底层具体实现，而不是依赖抽象。
- 在提出重构建议时，需说明 *为什么* 该调整能够改善内聚性或降低耦合度，并给出一个最小且安全的拆分方案。
- 如果重构规模较大，应提供一个渐进式改造计划，而不是一次性的大规模重写。

### 3) vLLM NPU平台扩展规范

- 加载`references/architecture.md`，获取具体的检查提示
- 重点关注以下方面：
     **层级放置规范 (Directory Layout)**：所有新增代码必须严格按组件类型归档：NPU 适配层存放在 `layers/` 或 `v1/layers/`，Attention 实现放入 `attention/backends/`，模型实现放入 `v1/models/`，补   件按通用或模型维度存放在 `vllm_patches/patches/` 下。
     **插件入口注册 (Plugin Entry Points)**：所有功能扩展必须通过 `pyproject.toml` 的 `entry-points` 进行声明式注册（包括 `vllm.platform` 和 `vllm.plugins`）。严禁在非项目入口处硬编码插件加载逻辑。
     **平台插件模式 (Platform Plugin)**： `NPUPlatform` 类必须继承自 `Platform` 基类。重点检查 `get_device_name` 是否锁定为 `"npu"`，以及 `get_attn_backend_cls` 是否根据硬件特性动态返回了正确的   力机制后端类。
     **补丁管理机制 (Patch Manager)**：严禁直接修改 vLLM 源码。必须继承 `VLLMPatch` 并使用 `@register_patch` 装饰器。补丁应用需受 `OMNI_NPU_VLLM_PATCHES` 环境变量控制，且 `_attr_names_to_apply`   须精准覆盖目标方法。
     **CustomOp OOT 注册 (Out-Of-Tree Operators)**：针对 `layernorm`、`linear` 等算子适配，必须使用 `@BaseClass.register_oot` 模式。实现类须继承 vLLM 基类并实现 `forward_oot` 方法，确保参数签名   类完全一致。
     **模型与连接器注册 (Registry Pattern)**：新模型需在 `v1/models/__init__.py` 中通过 `ModelRegistry.register_model` 挂载；KV Connector 需使用 `_safe_register` 注册。所有注册必须包含清晰的字    标识符与对应的模块路径。
     **量化配置准则 (Quantization Pattern)**：自定义量化方案必须通过 `@register_quantization_config` 注册。架构要求：禁止对量化逻辑使用 Patch，必须通过继承开源 vLLM 已有的量化基类（如 `CompressedTensorsConfig`）来实现 NPU 特化逻辑。

### 4) 代码风格检查

- 加载`references/code_style_checklist.md`，获取具体的检查提示
- 重点关注以下方面：
    - **文件头与版权声明 (License Header)**：所有 Python 文件必须包含 `SPDX-License-Identifier: MIT` 及华为版权声明。严禁漏掉版权年份或公司全称。
    - **严格的导入顺序 (Import Sorting)**：导入需分三组并以空行分隔：1. 标准库；2. 第三方库（`torch`, `vllm` 等）；3. 本地 `omni_npu` 模块。严禁跨组混排。
    - **命名一致性 (Naming Standards)**：类名使用 `PascalCase`；函数、方法和变量使用 `snake_case`；常量使用 `UPPER_SNAKE_CASE`。私有成员必须以 `_` 开头。
    - **强制类型提示 (Explicit Typing)**：所有公共接口、函数参数及返回值必须标注类型提示。对于复杂类型，需从 `typing` 模块导入 `Optional`, `List`, `Dict` 等。
    - **Google 风格文档字符串 (Docstrings)**：类和函数必须包含文档字符串。函数文档需采用 Google 风格，明确标注 `Args:` 和 `Returns:` 块。
    - **错误处理与日志规范 (Error Handling)**：内部逻辑校验使用 `assert`；环境适配或外部调用使用 `try-except`。记录状态必须使用 `vllm.logger.init_logger`，严禁直接使用原生 `print`。
    - **代码布局与空行 (Code Layout)**：顶级类/函数定义之间保留两个空行。类内部结构顺序为：类属性 -> `__init__` -> 公有方法 -> 私有方法（`_`）。
    - **行宽与折行 (Line Length & Wrapping)**：单行长度限制为 100 个字符。超过限制的函数调用、列表或字典定义必须进行多行拆分，确保视觉整洁。
    - **引号与空格 (Quotes & Spacing)**：普通字符串统一使用双引号 `"`；字典的键（Keys）使用单引号 `'`。二元操作符两端必须加空格（除简单的乘法运算外）。

### 5) NPU适配特定的设计原则

- 加载`references/npu_patterns.md`，获取具体的检查提示
- 重点关注以下方面：
    - **设备管理与可用性检查 (Device Management)**：必须使用 `current_platform.device_type` 动态获取设备类型，严禁硬编码 `npu:id`；在核心逻辑运行前，需通过 `hasattr(torch, "npu")` 校验 NPU 环境是否就绪。
    - **NPU 算子替换(NPU Operators)**：优先使用 `torch_npu` 提供的特化融合算子（如 `npu_swiglu` 替代 `SiluAndMul`，`npu_rms_norm` 替代通用归一化）以实现极致性能，并确保 RoPE 等操作使用 NPU 特有的 `TND` 布局。
    - **分布式通信与 HCCL (Communication)**：分布式并行操作应通过 `get_tp_group().device_group` 执行，确保底层通信后端正确由 NCCL 切换至 HCCL，并使用 `NPUCommunicator` 进行语义对齐。
    - **关键环境变量配置 (Environment Variables)**：必须正确响应并处理 `ASCEND_RT_VISIBLE_DEVICES`（设备可见性）、`TORCH_COMPILE_GE`（图编译开关）以及 `OMNI_NPU_VLLM_PATCHES`（补丁控制）等运行时环境变量。
    - **性能优化 (Performance Optimization)**：针对NPU平台，需要遵循一些特定的原则以便提升模型推理性能。
    - **特定架构设计原则 (Special Architecture Design)**：严格区分 `layers`（Base layer）与 `v1/layers`（高性能layer）；新增模块必须通过插件/注册表机制扩展，避免直接对 vLLM 源码打补丁；自定义模型脚本必须有引用`v1/layers`。

### 6) 输出格式

如果发现问题，评估结论只能是`REQUEST_CHANGES`
请按以下结构组织你的评审结果：

```markdown
## 代码评审 By AI Assistant

**已评审文件**：X 个文件，变更 Y 行  
**总体评估**：[APPROVE / REQUEST_CHANGES / COMMENT]

---
## 代码变更总结
按功能模块总结主要的变化点

## 发现的问题

### P0 - 严重
（无或列出问题）

### P1 - 高
1. **[file:line]** 问题简述
  - 问题描述
  - 修复建议

### P2 - 中
2. （在各级别之间持续编号）
  - ...

### P3 - 低
...

---

## 评审结论
- 如果发现问题，结论为"建议修改后再提交"
- 如果未发现问题，结论为"同意合入"

```

## 参考资料

详细实现示例可参考：

- [solid_checklist.md](references/solid_checklist.md) - SOLID 设计坏味道识别提示
- [architecture.md](references/architecture.md) - vLLM NPU平台扩展规范
- [code_style_checklist.md](references/code_style_checklist.md) - 代码风格检查
- [npu_patterns.md](references/npu_patterns.md) - NPU平台适配特定设计原则