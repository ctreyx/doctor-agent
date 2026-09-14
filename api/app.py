r"""FastAPI 应用入口（最小可用版）。

启动：
    & ".\.venv\Scripts\python.exe" -m uvicorn api.app:app --reload --port 8000

访问：
    http://127.0.0.1:8000/health   —— 健康检查
    http://127.0.0.1:8000/docs     —— 自动生成的交互式 API 文档
"""
from dotenv import load_dotenv

# 必须最先执行：doctor_agent.tools 在导入时就会用 TAVILY_API_KEY 初始化联网工具，
# 若环境变量未就绪，导入阶段即报错（与 main.py 的 load_dotenv 同理）。
load_dotenv()

from fastapi import FastAPI
from api.routers import auth, chat            # ← 加在顶部 import 区

app = FastAPI(
    title="doctor-agent API",
    version="0.1.0",
    description="医疗问答 RAG Agent 的 HTTP 接口层",
)


@app.get("/")
async def root():
    """根路径：给出可用入口提示。"""
    return {
        "service": "doctor-agent",
        "status": "running",
        "health": "/health",
        "docs": "/docs",
    }


app.include_router(chat.router, prefix="/api/v1", tags=["chat"])   #  对话

app.include_router(auth.router, prefix="/api/v1", tags=["auth"])   # ← 加这行


@app.get("/health")
async def health():
    """健康检查：供负载均衡 / K8s 探活使用。"""
    return {"status": "ok", "service": "doctor-agent"}
