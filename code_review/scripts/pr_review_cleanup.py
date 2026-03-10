import subprocess
from pathlib import Path
import sys

def cleanup(pr_num):
    repo_dir = Path.cwd()
    worktree_dir = repo_dir.parent / f"pr{pr_num}_review_zone"
    pr_branch = f"pr-{pr_num}-head"

    print(f"==> [3/3] 正在清理环境...")

    if worktree_dir.exists():
        subprocess.run(f"git worktree remove -f \"{worktree_dir}\"", shell=True, capture_output=True)
    
    subprocess.run(f"git branch -D {pr_branch}", shell=True, capture_output=True)
    print("✨ 环境已恢复原状。")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pr_3_cleanup.py <PR_NUM>")
    else:
        cleanup(sys.argv[1])