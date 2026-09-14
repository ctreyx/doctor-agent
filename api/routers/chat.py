"""对话接口。"""
from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from api.schemas import ChatRequest, ChatResponse
from api.services.agent import ask

from api.deps import verify_token



router = APIRouter()

@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(verify_token)],   # ← 加这一行：无 token 直接 401
)
# @router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # graph.invoke 是同步阻塞的，必须丢线程池，否则会卡住事件循环
    out = await run_in_threadpool(ask, req.message, req.thread_id)
    r = out["result"]
    return ChatResponse(
        thread_id=out["thread_id"],
        answer=r.get("generation", ""),
        query=r.get("query", ""),
        cache_hit=r.get("cache_hit", False),
        cache_reason=r.get("cache_reason", ""),
    )




