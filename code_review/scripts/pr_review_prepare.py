import subprocess
from pathlib import Path
import sys

def run_cmd(cmd, cwd=None):
    res = subprocess.run(cmd, shell=True, text=True, encoding='utf-8', 
                         errors='ignore', cwd=cwd, capture_output=True)
    return res

def prepare_env(pr_num, base_branch="master"):
    repo_dir = Path.cwd()
    pr_branch = f"pr-{pr_num}-head"
    worktree_dir = repo_dir.parent / f"pr{pr_num}_review_zone"

    print(f"==> [1/3] 准备环境: PR #{pr_num}")

    # 1. 获取最新代码
    run_cmd(f"git fetch origin pull/{pr_num}/head:{pr_branch}")
    run_cmd(f"git fetch origin {base_branch}")

    # 2. 创建隔离工作区
    if worktree_dir.exists():
        subprocess.run(f"git worktree remove -f \"{worktree_dir}\"", shell=True)
    
    print(f"创建临时工作区: {worktree_dir}")
    run_cmd(f"git worktree add \"{worktree_dir}\" origin/{base_branch}")

    # 3. 模拟合并 (核心：--no-commit --no-ff)
    print(f"正在模拟合并 PR 分支到 {base_branch}...")
    merge_res = run_cmd(f"git merge --no-commit --no-ff {pr_branch}", cwd=str(worktree_dir))

    if merge_res.returncode != 0:
        print("\n" + "!"*40)
        print("发现合并冲突！请手动进入工作区解决：")
        print(f"cd {worktree_dir}")
        print("解决冲突并 git add 后，再运行分析脚本。")
        print("!"*40 + "\n")
        return False, worktree_dir

    print("✅ 模拟合并成功，变更已进入暂存区 (Staging Area)。")
    return True, worktree_dir

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pr_1_prep.py <PR_NUM> [BASE]")
    else:
        prepare_env(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "master")