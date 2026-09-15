"""FastAPI 专用图实例：异步 checkpointer（支持流式 + 多轮记忆）。

与 main.py 的 graph 分开：那个给 LangGraph Server 用，这个给 FastAPI 用。

为什么用 AsyncSqliteSaver？
    流式接口走 graph.astream()（异步），同步 SqliteSaver 不支持异步方法
    （会抛 NotImplementedError）。

为什么在 lifespan 里初始化？
    aiosqlite 连接必须在事件循环中创建，模块级 import 时还没有事件循环。
    因此这里只暴露 startup() / shutdown() / get_graph()，由 app 的 lifespan 调度。
"""
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from doctor_agent.graph import build_graph

_conn: aiosqlite.Connection | None = None
_graph = None


async def startup() -> None:
    """应用启动时调用（在 FastAPI lifespan 内）。"""
    global _conn, _graph
    _conn = await aiosqlite.connect("resources/api_checkpoint.db")
    _graph = build_graph(checkpointer=AsyncSqliteSaver(_conn))


async def shutdown() -> None:
    """应用关闭时释放数据库连接。"""
    global _conn, _graph
    if _conn is not None:
        await _conn.close()
    _conn = None
    _graph = None


def get_graph():
    """获取图实例（须先执行 startup）。"""
    if _graph is None:
        raise RuntimeError("graph 未初始化：请确认 FastAPI lifespan 已调用 startup()")
    return _graph
