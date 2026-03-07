---
name: gitee_approve_pr
description: 专门用于在 Gitee 平台上对指定的 Pull Request (PR) 提交“审查通过 (Approved)”的操作。当用户表达“同意”、“通过”、“LGTM”或“过审”意图时使用。
---

# 🤖 Claude Skill: Gitee PR Approval

当你收到包含 "同意这个PR"、"Approve this PR" 或 "LGTM" 的指令时，请按照以下逻辑操作：

### 1. 识别参数
- **Owner/Repo**: 从当前项目目录或 git 远程仓库配置中识别。
- **PR Number**: 从用户指令中提取。如果用户没给编号，请尝试从对话上下文或最新的 PR 列表中查找。

### 2. 环境验证
- 检查是否存在环境变量 `GITEE_TOKEN`。如果缺失，请告知用户。

### 3. 执行动作
- 使用python执行脚本：`skills/scripts/pr_approve.py <owner> <repo> <number>`

### 4. 成功反馈
- 脚本执行成功后，向用户确认：“已通过 API 为 [Owner/Repo] 的第 [Number] 号 PR 提交了‘审查通过’。”