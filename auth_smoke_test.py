"""鉴权冒烟测试：验证「无 token 401 / 登录发 token / 带 token 200」。

用法（先启动后端）：
    & ".\.venv\Scripts\python.exe" auth_smoke_test.py
"""
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8001"


def post(path: str, payload: dict, token: str | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")[:120]


print("1) 无 token 调 /chat（预期 401）")
code, body = post("/api/v1/chat", {"message": "测试", "thread_id": "auth-1"})
print(f"   -> {code} {body}\n")

print("2) 错误密码登录（预期 401）")
code, body = post("/api/v1/auth/login", {"username": "admin", "password": "wrong"})
print(f"   -> {code} {body}\n")

print("3) 正确登录（预期 200 + token）")
code, body = post("/api/v1/auth/login", {"username": "admin", "password": "admin@123"})
print(f"   -> {code} {body}")
token = body.get("access_token") if isinstance(body, dict) else None
print(f"   token = {token}\n")

print("4) 带上正确 token 调 /chat（预期 200）")
code, body = post(
    "/api/v1/chat",
    {"message": "肺癌早期有哪些症状？", "thread_id": "auth-1"},
    token=token,
)
if isinstance(body, dict):
    print(f"   -> {code} | answer={body.get('answer', '')[:40]}...")
else:
    print(f"   -> {code} {body}")
