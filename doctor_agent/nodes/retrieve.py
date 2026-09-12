"""检索节点与路由。

对外接口：
    ``retrieve_node``  混合检索 + 重排，把结果与分数写回 state；
    ``route``          条件路由：分数达标 → 生成，否则 → 联网兜底。
"""
from doctor_agent import config
from doctor_agent.retrieval import format_context, get_retrieval, rerank_documents
from doctor_agent.retrieval_cache import get_cached, set_cached
from doctor_agent.state import State
from doctor_agent.tools import latest_user_query


def retrieve_node(state: State) -> dict:
    """节点：混合检索 + rerank，把结果、分数、实际查询写回 state。

    查询优先取 ``state["query"]``（改写节点产出），
    缺失时回退到最新用户消息——保证即使跳过 rewrite 也能检索。

    高频问题命中缓存时直接返回上次的 ``(context, score)``，跳过整套检索；
    HyDE 二次检索的 query 是每次不稳定的假设答案，不参与缓存。

    缓存可观测性：无论命中与否，均把 ``cache_hit / cache_reason / cache_mode``
    写回 state，使 LangSmith 链路追踪可直接看到本次检索的缓存状态。
    """
    query = state.get("query") or latest_user_query(state["messages"])
    is_hyde = state.get("hyde_done", False)
    cache_mode = "on" if config.RAG_CACHE_ENABLED else "off"

    if not is_hyde:
        cached = get_cached(query)
        if cached is not None:
            context, score, reason = cached
            print(f"[cache] 命中({reason})：{query[:30]}… score={score:.3f}")
            return {
                "context": context,
                "score": score,
                "query": query,
                "cache_hit": True,
                "cache_reason": reason,
                "cache_mode": cache_mode,
            }

    ensemble, _ = get_retrieval()
    ranked = rerank_documents(query, ensemble.invoke(query))
    score = ranked[0][0] if ranked else 0.0
    context = format_context([d for _, d in ranked])

    if not is_hyde:
        set_cached(query, context, score)

    if is_hyde:
        reason = "bypass_hyde"
    elif not config.RAG_CACHE_ENABLED:
        reason = "cache_disabled"
    else:
        reason = "miss"

    return {
        "context": context,
        "score": score,
        "query": query,   # 把实际使用的 query 写回 state，供测试与后续节点复用
        "cache_hit": False,
        "cache_reason": reason,
        "cache_mode": cache_mode,
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
