"""联网兜底节点。

本地检索分数不达标（低于 ``RAG_MIN_SCORE``）时，调用 Tavily 联网搜索；
联网失败则回退到本地检索结果，保证链路不中断。
"""
from doctor_agent.state import State
from doctor_agent.tools import format_web_result, latest_user_query, tavily_tool


def web_search_node(state: State) -> dict:
    """节点：RAG 不够时，Tavily 联网兜底。"""
    query = state.get("query") or latest_user_query(state["messages"])
    try:
        raw = tavily_tool.invoke({"query": query})
        return {"context": format_web_result(raw)}
    except Exception:
        # 联网失败 → 退回本地结果，避免整条链路失败
        return {"context": state.get("context", "当前暂无相关参考资料。")}
