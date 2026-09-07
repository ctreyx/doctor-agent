"""混合检索与重排。

对外接口：
    ``get_retrieval()``   惰性单例，返回 ``(EnsembleRetriever, CrossEncoder)``；
    ``rerank_documents()`` 用 CrossEncoder 对候选做精排；
    ``format_context()``   按 system prompt 约定格式拼装检索结果。

检索策略：
    Pinecone 向量检索 + BM25 关键词检索，经 ``EnsembleRetriever`` 融合
    （权重见 ``config.ENSEMBLE_WEIGHTS``），再由 CrossEncoder（Sigmoid 激活）
    把 logits 映射到 0~1 概率分做精排。
"""
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import CrossEncoder
from torch import nn

from doctor_agent import config
from doctor_agent.document_loader import get_documents
from doctor_agent.embeddings import get_embeddings
from doctor_agent.vector_store import get_index

# 惰性单例：CrossEncoder 下载 + BM25 建索引都很重，
# 首次调用才初始化，避免 Server 启动时卡住。
_retrieval = None


def get_retrieval():
    """返回 ``(混合检索器, 重排器)`` 的惰性单例。"""
    global _retrieval
    if _retrieval is None:
        # CrossEncoder 重排器：Sigmoid 把 logits(-10~10) 映射为概率(0~1)
        reranker = CrossEncoder(
            config.RERANK_MODEL_NAME,
            activation_fn=nn.Sigmoid(),
        )

        # 向量检索 + BM25 关键词检索 → 加权融合
        vs = PineconeVectorStore(index=get_index(), embedding=get_embeddings())
        vector_retriever = vs.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
        bm25_retriever = BM25Retriever.from_documents(get_documents())
        bm25_retriever.k = config.RETRIEVAL_K
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=config.ENSEMBLE_WEIGHTS,
        )
        _retrieval = (ensemble, reranker)
    return _retrieval


def rerank_documents(query: str, docs: list[Document], top_n: int = config.RERANK_TOP_N):
    """CrossEncoder 精排，返回 ``[(score, doc)]`` 按分数降序，最多 ``top_n`` 条。"""
    if not docs:
        return []
    _, reranker = get_retrieval()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return ranked[:top_n]


def format_context(docs: list[Document]) -> str:
    """按 ``prompts/doctor_prompt.md`` 约定的格式组织检索结果。"""
    if not docs:
        return "当前暂无相关参考资料。"
    lines = ["【检索结果开始】"]
    for i, doc in enumerate(docs, start=1):
        dept = doc.metadata.get("department", "未知科室")
        title = doc.metadata.get("title", "")
        lines.append(f"[{i}] 来源：{dept} | 标题：{title} | 内容：{doc.page_content}")
    lines.append("【检索结果结束】")
    return "\n".join(lines)
