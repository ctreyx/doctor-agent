"""Pinecone 向量库：初始化、ID 生成与增量写入。

对外接口：
    ``get_index()``        惰性单例，首次调用才连接 / 创建索引；
    ``upsert_documents()`` 增量写入（只处理新 id 或内容变化的 chunk）。
"""
import os

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from doctor_agent import config
from doctor_agent.document_loader import doc_hash, load_manifest, save_manifest
from doctor_agent.embeddings import get_embeddings
from doctor_agent.retrieval_cache import invalidate_cache

_pc: Pinecone | None = None
_index = None


def get_index():
    """返回 Pinecone 索引的惰性单例。

    首次调用才真正建立连接；索引不存在时按配置自动创建。
    维度必须与 embedding 输出一致（见 ``config.EMBEDDING_DIMENSION``）。
    """
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
        if config.PINECONE_INDEX_NAME not in _pc.list_indexes().names():
            _pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=config.EMBEDDING_DIMENSION,
                metric=config.PINECONE_METRIC,
                spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
            )
        _index = _pc.Index(config.PINECONE_INDEX_NAME)
    return _index


def build_ids(docs) -> list[str]:
    """生成稳定、幂等的文档 ID：``zhongliu-{row_id}`` 或 ``zhongliu-{row_id}-{chunk_id}``。

    稳定 ID 是增量写入的前提——同一条数据多次运行不会产生重复向量。
    """
    return [
        f"zhongliu-{d.metadata['row_id']}"
        + (f"-{d.metadata['chunk_id']}" if "chunk_id" in d.metadata else "")
        for d in docs
    ]


def upsert_documents(docs, batch_size: int = 100) -> int:
    """增量写入向量库，返回本次实际写入条数。

    仅处理「新 id」或「内容变化（doc_hash 不同）」的 chunk，
    依赖 ``cache/index_manifest.json`` 记录每个 id 的内容指纹，支持断点续跑。
    """
    index = get_index()
    manifest = load_manifest()

    to_upsert = [
        (i, d, doc_hash(d))
        for i, d in zip(build_ids(docs), docs)
        if manifest.get(i) != doc_hash(d)
    ]
    if not to_upsert:
        return 0

    vectorstore = PineconeVectorStore(index=index, embedding=get_embeddings())
    for start in range(0, len(to_upsert), batch_size):
        batch = to_upsert[start:start + batch_size]
        vectorstore.add_documents(
            documents=[d for _, d, _ in batch],
            ids=[i for i, _, _ in batch],
            batch_size=batch_size,
        )
        for i, _, h in batch:
            manifest[i] = h
        save_manifest(manifest)   # 每批落盘，断点续跑

    # 缓存
    invalidate_cache()

    return len(to_upsert)
