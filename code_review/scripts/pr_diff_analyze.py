import subprocess
import json
import re
from pathlib import Path
import sys

def run_cmd(cmd, cwd=None):
    res = subprocess.run(cmd, shell=True, text=True, encoding='utf-8', errors='ignore', cwd=cwd, capture_output=True)
    return res.stdout.strip() if res.returncode == 0 else None

def parse_diff_by_mode(diff_str, mode='added'):
    lines = []
    cur_old, cur_new = 0, 0
    if not diff_str: return lines
    
    for dl in diff_str.split('\n'):
        if dl.startswith('@@'):
            m = re.search(r'\-([0-9]+).*?\+([0-9]+)', dl)
            if m: cur_old, cur_new = map(int, m.groups())
        elif dl.startswith('+') and not dl.startswith('+++'):
            if mode == 'added': lines.append({"line": cur_new, "code": dl[1:]})
            cur_new += 1
        elif dl.startswith('-') and not dl.startswith('---'):
            if mode == 'deleted': lines.append({"line": cur_old, "code": dl[1:]})
            cur_old += 1
        elif dl.startswith(' '):
            if mode == 'context': lines.append({"line": cur_new, "code": dl[1:] or " "})
            cur_old += 1
            cur_new += 1
    return lines

def analyze_staged(pr_num, worktree_dir, context_len=5):
    print(f"==> [2/3] 正在分析差异，结果将存入工作区...")
    
    worktree_path = Path(worktree_dir)
    files_output = run_cmd("git diff --name-status --cached -M", cwd=str(worktree_path))
    if not files_output:
        print("暂存区没有发现任何变更。")
        return

    all_data = []
    for line in files_output.split('\n'):
        parts = line.split()
        if not parts: continue
        status, file_path = parts[0], parts[-1]
        
        item = {"file": file_path, "status": status, "added": [], "deleted": [], "context": []}
        d0 = run_cmd(f"git diff --cached -U0 -- \"{file_path}\"", cwd=str(worktree_path))
        dctx = run_cmd(f"git diff --cached -U{context_len} -- \"{file_path}\"", cwd=str(worktree_path))

        item["added"] = parse_diff_by_mode(d0, 'added')
        item["deleted"] = parse_diff_by_mode(d0, 'deleted')
        item["context"] = parse_diff_by_mode(dctx, 'context')
        all_data.append(item)

    # 修改点：将路径指向 worktree_path
    output_file = worktree_path / f"pr_{pr_num}_review.json"
    
    raw_json = json.dumps(all_data, indent=2, ensure_ascii=False)
    compact_json = re.sub(
        r'\{\s+"line":\s+(\d+),\s+"code":\s+(.*?)\n\s+\}',
        r'{"line": \1, "code": \2}',
        raw_json, flags=re.DOTALL
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(compact_json)
    print(f"✅ 分析完成！JSON 已生成在工作区: {output_file}")

if __name__ == "__main__":
    analyze_staged(sys.argv[1], sys.argv[2])