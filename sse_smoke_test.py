"""SSE 流式接口测试（用 Python，避开 PowerShell 的引号/编码坑）。

用法（先启动后端）：
    & ".\.venv\Scripts\python.exe" sse_smoke_test.py
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8001"
TOKEN = "fake-token-doctor-agent-2026"

payload = json.dumps(
    {"message": "肺癌早期有哪些症状？", "thread_id": "sse-1"},
    ensure_ascii=False,
).encode("utf-8")

req = urllib.request.Request(
    f"{BASE}/api/v1/chat/stream",
    data=payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOKEN}",
    },
    method="POST",
)

print("=== 流式输出 ===")
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        print(f"[HTTP {resp.status}] content-type={resp.headers.get('content-type')}\n")
        token_count = 0
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue

            if "token" in obj:
                token_count += 1
                print(obj["token"], end="", flush=True)
            elif obj.get("done"):
                print("\n\n=== done ===")
                print(json.dumps(obj, ensure_ascii=False, indent=2))
                print(f"\n共收到 {token_count} 个 token 事件")
            elif "error" in obj:
                print("\n=== error ===")
                print(obj["error"])
except urllib.error.HTTPError as e:
    print(f"[HTTP {e.code}] {e.read().decode('utf-8')[:300]}")
