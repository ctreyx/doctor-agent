"""对话接口（同步 + 流式 SSE）。"""
import json
from uuid import uuid4
import logging
from fastapi.responses import StreamingResponse

from api.deps import verify_token
from api.graph_instance import get_graph
from api.schemas import ChatRequest, ChatResponse
from api.services.agent import ask
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage
router = APIRouter()
from doctor_agent import config


logger = logging.getLogger(__name__)     # name = "api.routers.chat"


def _sse(payload: dict, event: str | None = None) -> str:
    """SSE 消息；event 非空时带上 event 字段。"""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------- 同步接口（保持不变） ----------
@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_token)])
async def chat(req: ChatRequest):
    out = await ask(req.message, req.thread_id)
    r = out["result"]
    return ChatResponse(
        thread_id=out["thread_id"],
        answer=r.get("generation", ""),
        query=r.get("query", ""),
        cache_hit=r.get("cache_hit", False),
        cache_reason=r.get("cache_reason", ""),
    )


# ---------- 流式接口（新增） ----------
@router.post("/chat/stream", dependencies=[Depends(verify_token)], summary="流式对话（SSE）")
async def chat_stream(req: ChatRequest):
    """SSE 流式返回，逐 token 推送。

    事件约定（每条都是 JSON）：
        {"token": "..."}                                          —— 增量文本
        {"done": true, "thread_id": ..., "query": ..., "cache_hit": ..., "cache_reason": ...}  —— 结束
        {"error": "..."}                                          —— 出错
    """
    tid = req.thread_id or str(uuid4())
    run_config = {"configurable": {"thread_id": tid}}
    graph = get_graph()



    # ---------- 续写：前置校验（读 checkpoint，不信客户端） ----------
    if req.resume:
        if not req.thread_id:
            raise HTTPException(status_code=400, detail="续写必须携带 thread_id")

        snap = await graph.aget_state(run_config)          # ★ 读上轮 state
        prev = snap.values or {}

        if not prev.get("messages"):
            raise HTTPException(status_code=409, detail="该会话没有可续写的内容")
        if not prev.get("truncated"):
            raise HTTPException(status_code=409, detail="上一轮回答未被截断，无需续写")

        done = prev.get("continue_count", 0) or 0
        if done >= config.CONTINUE_MAX_ROUNDS:
            raise HTTPException(
                status_code=429,
                detail=f"本轮最多续写 {config.CONTINUE_MAX_ROUNDS} 次",
            )

        graph_input = {
            "messages": [HumanMessage(content=config.CONTINUE_PROMPT)],
            "resume": True,               # ★ 让 route_entry 走 generate 捷径
            "continue_count": done + 1,
        }
    else:
        graph_input = {
            "messages": [HumanMessage(content=req.message)],
            "resume": False,              # ★ 正常链路
            "continue_count": 0,
        }

    async def event_gen():
        final_state: dict = {}
        truncated = False
        char_count = 0
        try:
            async for mode, payload in graph.astream(
                graph_input,
                config=run_config,
                stream_mode=["messages", "values"],
            ):
                if mode == "messages":
                    chunk, meta = payload
                    if meta.get("langgraph_node") != "generate":
                        continue

                    if chunk.content:
                        char_count += len(chunk.content)
                        yield _sse({"token": chunk.content})

                    # ★ max_length 检测截断 如果输出太多
                    finish = (chunk.response_metadata or {}).get("finish_reason")
                    if finish == "length" and not truncated:      # ★ 只发一次
                        truncated = True
                        yield _sse(
                            {"reason": "length", "chars": char_count},
                            event="max_length",          # ← 独立事件
                        )
                elif mode == "values":
                    final_state = payload

            yield _sse({
                "done": True,
                "thread_id": tid,
                "query": final_state.get("query", ""),
                "cache_hit": final_state.get("cache_hit", False),
                "cache_reason": final_state.get("cache_reason", ""),
                "truncated": truncated,                      # ← done 里冗余带一份
                  "continue_count": final_state.get("continue_count", 0),

            })

        except Exception as e:
            # 流一旦开始，HTTP 状态码已发出，只能推错误事件收尾
            # yield _sse({"error": f"{type(e).__name__}: {e}"})
            logger.exception("chat_stream failed thread_id=%s", tid)
            yield _sse({"error": "服务暂时不可用，请稍后重试", "trace_id": tid})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # 让 Nginx 等反代不缓冲，否则流式会被"攒着一起吐"
        },
    )
