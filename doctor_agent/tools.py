"""Agent 工具与消息辅助函数。

对外接口：
    ``search_medical_knowledge``  本地医学知识库检索工具（``@tool``）；
    ``tavily_tool``              Tavily 联网搜索工具；
    ``latest_user_query()``      取最近一条用户消息；
    ``format_web_result()``      给联网结果包一层来源标注。
"""
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from doctor_agent import config
from doctor_agent.retrieval import format_context, get_retrieval, rerank_documents


def format_web_result(raw: str) -> str:
    """给 Tavily 返回结果包一层来源标注，便于与本地检索结果区分。"""
    return "【联网搜索结果开始】\n" + raw + "\n【联网搜索结果结束】"


def latest_user_query(messages: list) -> str:
    """取最近一条用户消息（跳过 AI / 工具消息）。

    倒序遍历，返回第一条 ``HumanMessage`` 的内容；
    多轮追问场景下，这保证拿到的是「用户最新的追问」而非历史回答。
    """
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content
    return messages[-1].content if messages else ""


@tool
def search_medical_knowledge(query: str) -> str:
    """检索本地医学知识库。

    当用户询问疾病症状、诊断、治疗、用药、康复、预防等医学问题时，
    调用此工具获取基于权威医疗数据库的参考资料。
    返回结果会包含来源与内容，请严格基于这些内容回答。"""
    ensemble, _ = get_retrieval()
    candidates = ensemble.invoke(query)                  # 向量 + BM25 混合检索
    top = rerank_documents(query, candidates)            # CrossEncoder 重排
    return format_context([d for _, d in top])


tavily_tool = TavilySearch(
    max_results=config.TAVILY_MAX_RESULTS,
    topic="general",
)
