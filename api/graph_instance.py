"""FastAPI 专用图实例：带 checkpointer，支持多轮记忆。

与 main.py 的 graph 分开：那个给 LangGraph Server 用（Server 自己注入），
这个给 FastAPI 用（自己管），避免两套 checkpointer 打架。
"""
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from doctor_agent.graph import build_graph

# check_same_thread=False：FastAPI 把同步代码丢线程池跑，必须关掉线程检查
_conn = sqlite3.connect("resources/api_checkpoint.db", check_same_thread=False)
checkpointer = SqliteSaver(_conn)

graph = build_graph(checkpointer=checkpointer)


