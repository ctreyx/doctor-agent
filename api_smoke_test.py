"""FastAPI 冒烟测试：验证 /health 与多轮记忆 /chat。

用法（先在另一个终端启动服务）：
    & ".\.venv\Scripts\python.exe" api_smoke_test.py
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8001"


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


print("1) /health ->", get("/health"))

r1 = post("/api/v1/chat", {"message": "化疗后恶心怎么办？", "thread_id": "smoke-1"})
print("\n2) 第 1 轮")
print("   thread_id =", r1.get("thread_id"))
print("   query     =", r1.get("query"))
print("   answer    =", (r1.get("answer") or "")[:60])
print("   cache     =", r1.get("cache_hit"), r1.get("cache_reason"))

r2 = post("/api/v1/chat", {"message": "那会持续多久？", "thread_id": "smoke-1"})
print("\n3) 第 2 轮（残句追问，依赖历史）")
print("   query     =", r2.get("query"))
print("   answer    =", (r2.get("answer") or "")[:60])

hit = "恶心" in (r2.get("query") or "")
print("\n结论：", "✅ 记忆生效（追问被消解为完整查询）" if hit else "❌ 记忆未生效")
