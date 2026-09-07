"""DashScope text-embedding-v3 封装。

为什么自实现而不是用 langchain-dashscope：
    官方 ``langchain-dashscope`` / ``langchain-community`` 的 dashscope 集成已停更，
    与 LangChain 1.x 不兼容，故直接基于官方 dashscope SDK 实现 ``Embeddings`` 接口，
    行为可控、无额外依赖。

对外接口：
    ``get_embeddings()`` —— 惰性单例，首次调用才实例化（避免 import 即初始化）。
"""
import os
from typing import List

from dashscope import TextEmbedding
from langchain_core.embeddings import Embeddings

from doctor_agent import config


class DashScopeTextEmbeddingV3(Embeddings):
    """阿里云 text-embedding-v3，实现 LangChain ``Embeddings`` 接口。"""

    def __init__(
        self,
        api_key: str,
        model: str = config.EMBEDDING_MODEL,
        dimension: int = config.EMBEDDING_DIMENSION,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
    ):
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size

    def _embed(self, texts: List[str], text_type: str) -> List[List[float]]:
        """分批调用 DashScope，返回向量列表。``text_type`` 为 query / document。"""
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            resp = TextEmbedding.call(
                model=self.model,
                input=texts[i:i + self.batch_size],
                dimension=self.dimension,
                text_type=text_type,
                api_key=self.api_key,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"embedding 失败: code={resp.code} msg={resp.message}")
            out.extend(item["embedding"] for item in resp.output["embeddings"])
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量编码文档（入库用）。"""
        return self._embed(texts, text_type="document")

    def embed_query(self, text: str) -> List[float]:
        """编码单条查询（检索用）。"""
        return self._embed([text], text_type="query")[0]


_embeddings: DashScopeTextEmbeddingV3 | None = None


def get_embeddings() -> DashScopeTextEmbeddingV3:
    """返回 embedding 实例的惰性单例。"""
    global _embeddings
    if _embeddings is None:
        _embeddings = DashScopeTextEmbeddingV3(
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        )
    return _embeddings
