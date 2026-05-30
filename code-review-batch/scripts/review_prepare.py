import subprocess
import json
import re
import os
import argparse
import sys
from pathlib import Path

def run_cmd(cmd, cwd=None):
    """通用命令执行，处理编码和错误"""
    try:
        # 使用 utf-8 编码，忽略无法解码的字符（如源码中的特殊注释）
        result = subprocess.check_output(
            cmd, shell=True, text=True, encoding='utf-8', errors='ignore', cwd=cwd
        )
        return result.strip()
    except Exception:
        return None

def get_pr_diff_json(pr_num, base_branch="master", context_len=5):
    repo_dir = Path.cwd()
    tmp_branch = f"pr-{pr_num}-tmp"
    worktree_dir = repo_dir.parent / f"pr{pr_num}_review"
    output_json = repo_dir / f"pr_{pr_num}_review.json"

    print(f"==> Starting PR #{pr_num} analysis task...")

    try:
        # --- 1. Git 环境准备 ---
        run_cmd("git config --global core.quotepath false")
        
        print(f"Fetching remote branch: pull/{pr_num}/head")
        if run_cmd(f"git fetch origin pull/{pr_num}/head:{tmp_branch}") is None:
            print(f"ERROR: Failed to fetch PR #{pr_num}. Please check the PR number or network.")
            return

        run_cmd(f"git fetch origin {base_branch}")

        # --- 2. 建立临时工作区 ---
        if worktree_dir.exists():
            run_cmd(f"git worktree remove -f \"{str(worktree_dir)}\"")
        
        print(f"Creating Worktree: {worktree_dir}")
        run_cmd(f"git worktree add \"{str(worktree_dir)}\" {tmp_branch}")

        mb = run_cmd(f"git merge-base origin/{base_branch} {tmp_branch}", cwd=str(repo_dir))
        if not mb:
            print("ERROR: Cannot find Merge-base ancestor.")
            return

        # --- 3. 分析文件变更 ---
        files_output = run_cmd(f"git diff --name-status {mb} {tmp_branch}", cwd=str(worktree_dir))
        if not files_output:
            print("No file changes detected.")
            return

        all_data = []
        for line in files_output.split('\n'):
            if not line.strip(): continue
            parts = line.split(None, 1)
            if len(parts) < 2: continue
            status, file_path = parts
            
            # 初始化文件结构
            file_item = {"file": file_path, "added_lines": [], "deleted_lines": [], "context": []}

            # 处理修改文件 (M)
            if status == 'M':
                # U0 获取纯变更行，U{context_len} 获取背景
                diff_u0 = run_cmd(f"git diff -U0 {mb} {tmp_branch} -- \"{file_path}\"", cwd=str(worktree_dir))
                diff_ctx = run_cmd(f"git diff -U{context_len} {mb} {tmp_branch} -- \"{file_path}\"", cwd=str(worktree_dir))

                # 提取变更行 (针对 HEAD 文件和 Merge-base 文件)
                cur_new = 0; cur_old = 0
                if diff_u0:
                    for dl in diff_u0.split('\n'):
                        if dl.startswith('@@'):
                            m = re.search(r'\-([0-9]+).*?\+([0-9]+)', dl)
                            if m: cur_old, cur_new = map(int, m.groups())
                        elif dl.startswith('+') and not dl.startswith('+++'):
                            file_item["added_lines"].append({"line": cur_new, "code": dl[1:]})
                            cur_new += 1
                        elif dl.startswith('-') and not dl.startswith('---'):
                            file_item["deleted_lines"].append({"line": cur_old, "code": dl[1:]})
                            cur_old += 1

                # 提取带行号的上下文 (HEAD 侧)
                # 每个 hunk 有自己的起始行号，需要重置
                if diff_ctx:
                    for cl in diff_ctx.split('\n'):
                        if cl.startswith('@@'):
                            m = re.search(r'\+([0-9]+)', cl)
                            if m:
                                # 重置为当前 hunk 的起始行号
                                ctx_new = int(m.group(1))
                        elif cl.startswith(' '):
                            file_item["context"].append({"line": ctx_new, "code": cl[1:] or " "})
                            ctx_new += 1
                        elif cl.startswith('+'):
                            # 跳过 added 行，但不添加到 context
                            ctx_new += 1

            # 处理新增文件 (A)
            elif status == 'A':
                full_path = worktree_dir / file_path
                if full_path.exists():
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            file_item["added_lines"] = [{"line": i+1, "code": l.rstrip()} for i, l in enumerate(f)]
                    except Exception: pass

            all_data.append(file_item)

        # --- 4. 生成并压缩 JSON (核心改进点) ---
        # 首先生成标准缩进的 JSON
        raw_json = json.dumps(all_data, indent=2, ensure_ascii=False)

        # 使用正则将 {"line": X, "code": "Y"} 内部的换行符全部抹掉
        # 此正则会匹配整个花括号对象，并将其压缩为一行
        compact_json = re.sub(
            r'\{\s+"line":\s+(\d+),\s+"code":\s+(.*?)\n\s+\}',
            r'{"line": \1, "code": \2}',
            raw_json, flags=re.DOTALL
        )

        with open(output_json, "w", encoding="utf-8") as f:
            f.write(compact_json)
        
        print(f"Success! Review JSON generated: {output_json}")

    except Exception as e:
        print(f"Exception: {e}")
    finally:
        # --- 5. 资源清理 (核心点) ---
        print("Cleaning up temporary resources...")
        os.chdir(str(repo_dir))
        if worktree_dir.exists():
            subprocess.run(f"git worktree remove -f \"{str(worktree_dir)}\"", shell=True, capture_output=True)
        subprocess.run(f"git branch -D {tmp_branch}", shell=True, capture_output=True)
        print("Cleanup completed. Environment restored.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gitee PR Analyzer for Claude Review")
    parser.add_argument("pr_number", type=int, help="PR 编号")
    parser.add_argument("--base", type=str, default="master", help="目标基准分支 (默认: master)")
    parser.add_argument("--ctx", type=int, default=5, help="上下文行数 (默认: 5)")
    
    args = parser.parse_args()
    get_pr_diff_json(args.pr_number, args.base, args.ctx)