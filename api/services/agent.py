"""Agent 服务层：封装 LangGraph 图调用。"""
from uuid import uuid4

from api.graph_instance import graph


def ask(message: str, thread_id: str | None = None) -> dict:
    """跑一轮对话。

    记忆要点：
        - thread_id 相同 → LangGraph 自动加载历史（多轮上下文）
        - 不传/新 id     → 全新会话
    """
    tid = thread_id or str(uuid4())
    result = graph.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": tid}},   # ← 记忆靠这一行
    )
    return {"thread_id": tid, "result": result}
