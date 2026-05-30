# Code Review Skills Directory

本目录包含用于 omni-infer 项目代码 review 的各种 skills。

## 概述

这些 skills 帮助团队对代码进行全面的 review，确保：
1. 代码符合 omni-infer 的风格规范
2. 代码架构设计与 vLLM 保持一致
3. Layer 模块注册符合规范（CustomOp/继承/量化等）

## Code Review 流程

```
发起 PR
   ↓
运行自动化评审
   ↓
给出给评审总结
   ↓
committer确认
   ↓
提交评审意见
   ↓
批准合并
```

## 使用指南

- 将`code-review/skills`拷贝到`.claude/skills`目录下
- 检视PR：
```shell
> /omni-npu-review 检视这个PR：https://gitee.com/omniai/omni-npu/pulls/xxx/files
```
- +1操作
通过关键词触发加分skill
```shell
> 同意/LGTM/审查通过
```

## 更新日志

- 2026-02-24: 重构文档结构，添加章节编号，新增 SOLID 检查清单
- 2026-02-13: 初始版本，创建 skills
- 2026-02-13: 移除 team_principles_check.md
- 2026-02-13: 添加 Layer 模块注册规范
- 2026-02-13: 更新 base layer vs 高性能 layer 说明
- 2026-02-13: 添加 patch 优先级原则（避免 patch，优先继承）

## 参考文档

- [`SKILL.md`](./SKILL.md) - 主技能定义，包含完整的审查流程和严重级别定义
- [`references/solid_checklist.md`](./references/solid_checklist.md) - SOLID 设计原则检查
- [`references/architecture.md`](./references/architecture.md) - vLLM NPU平台扩展规范
- [`references/code_style_checklist.md`](./references/code_style_checklist.md) - 代码风格检查
- [`references/npu_patterns.md`](./references/npu_patterns.md) - NPU平台适配特定设计原则