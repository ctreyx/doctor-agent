"""检索节点与路由。

对外接口：
    ``retrieve_node``  混合检索 + 重排，把结果与分数写回 state；
    ``route``          条件路由：分数达标 → 生成，否则 → 联网兜底。
"""
from doctor_agent import config
from doctor_agent.retrieval import format_context, get_retrieval, rerank_documents
from doctor_agent.state import State
from doctor_agent.tools import latest_user_query


def retrieve_node(state: State) -> dict:
    """节点：混合检索 + rerank，把结果、分数、实际查询写回 state。

    查询优先取 ``state["query"]``（改写节点产出），
    缺失时回退到最新用户消息——保证即使跳过 rewrite 也能检索。
    """
    query = state.get("query") or latest_user_query(state["messages"])
    ensemble, _ = get_retrieval()
    ranked = rerank_documents(query, ensemble.invoke(query))
    score = ranked[0][0] if ranked else 0.0
    return {
        "context": format_context([d for _, d in ranked]),
        "score": score,
        "query": query,   # 把实际使用的 query 写回 state，供测试与后续节点复用
    }


# def route(state: State) -> str:
#     """条件边：分数达标 → 直接生成；否则 → 联网。"""
#     return "generate" if state.get("score", 0.0) >= config.RAG_MIN_SCORE else "web_search"


def route(state: State) -> str:
    """条件边：分数达标 → 生成；分数低 → HyDE 二次检索；仍低 → 联网。

    决策顺序（HyDE 最多执行一次，避免死循环）：
        1. score ≥ RAG_MIN_SCORE            → generate
        2. score < RAG_MIN_SCORE 且未 HyDE → hyde（用假设答案再检索）
        3. score < RAG_MIN_SCORE 且已 HyDE → web_search
    """
    if state.get("score", 0.0) >= config.RAG_MIN_SCORE:
        return "generate"
    if config.HYDE_ENABLED and not state.get("hyde_done", False):
        return "hyde"
    return "web_search"
