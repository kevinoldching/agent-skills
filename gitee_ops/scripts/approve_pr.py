import sys
import os
import requests
import json
import urllib.parse

def approve_pr(owner, repo, number, comment):
    # --- 从环境变量读取配置 ---
    GITEE_ACCESS_TOKEN = os.environ.get("GITEE_ACCESS_TOKEN")
    if not GITEE_ACCESS_TOKEN:
        print("❌ 错误: 未设置环境变量 GITEE_ACCESS_TOKEN")
        sys.exit(1)

    # 代理配置 (可选)
    proxy_url = None
    if os.environ.get("HTTP_PROXY"):
        proxy_url = os.environ.get("HTTP_PROXY")
    elif os.environ.get("HTTP_PROXY_USER") and os.environ.get("HTTP_PROXY_PASSWORD"):
        user = os.environ.get("HTTP_PROXY_USER")
        password = os.environ.get("HTTP_PROXY_PASSWORD")
        host = os.environ.get("HTTP_PROXY_HOST", "127.0.0.1")
        port = os.environ.get("HTTP_PROXY_PORT", "8080")
        safe_user = urllib.parse.quote_plus(user)
        safe_pass = urllib.parse.quote_plus(password)
        proxy_url = f"http://{safe_user}:{safe_pass}@{host}:{port}"

    proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
    # ---------------

    url = f"https://gitee.com/api/v5/repos/{owner}/{repo}/pulls/{number}/review"
    payload = {
        "access_token": GITEE_ACCESS_TOKEN,
        "body": comment,
        "result": "pass"
    }
    
    try:
        # 使用 requests 处理网络请求，更稳健且易于调试
        response = requests.post(url, json=payload, proxies=proxies, timeout=10)
        if response.status_code in [200, 201]:
            print(f"✅ 审查已通过！PR #{number} 状态已更新。")
        else:
            print(f"❌ 失败。HTTP 状态码: {response.status_code}")
            print(f"详细错误: {response.text}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 网络连接异常: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("用法: python pr_approve.py <owner> <repo> <number> [comment]")
        sys.exit(1)
    
    comment = sys.argv[4] if len(sys.argv) > 4 else "LGTM! Approved via Claude Code."
    approve_pr(sys.argv[1], sys.argv[2], sys.argv[3], comment)