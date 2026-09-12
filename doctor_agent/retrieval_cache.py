import threading
import time
from collections import OrderedDict

from sentence_transformers import SentenceTransformer

from doctor_agent import config

# key -> (expire_ts, context, score, embedding)
_cache: OrderedDict[str, tuple] = OrderedDict()
_lock = threading.Lock()
_embedder = None
_hits = 0
_misses = 0


def _get_embedder() -> SentenceTransformer:
    """本地轻量 embedding（惰性加载，毫秒级、无网络）。"""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.RAG_CACHE_EMBED_MODEL)
    return _embedder


def normalize_query(query: str) -> str:
    if not query:
        return ""
    return " ".join(query.strip().lower().split())


def get_cached(query: str):
    """命中返回 ``(context, score, reason)``，否则 ``None``（过期条目惰性清理）。

    reason 取值：``exact``（精确键命中）/ ``semantic``（语义相似命中）。
    """
    global _hits, _misses
    if not config.RAG_CACHE_ENABLED:
        return None

    key = normalize_query(query)
    if not key:
        return None

    now = time.time()

    # ---- L1：精确 key 匹配（O(1)，零额外成本）----
    with _lock:
        item = _cache.get(key)
        if item is not None:
            expire_ts, context, score, _ = item
            if now <= expire_ts:
                _cache.move_to_end(key)
                _hits += 1
                return (context, score, "exact")
            del _cache[key]   # 过期惰性清理

    # ---- L2：本地 embedding 语义匹配（L1 miss 才触发）----
    q_emb = _get_embedder().encode(query, normalize_embeddings=True)
    best_key, best_sim, best_val = None, 0.0, None
    with _lock:
        for k, (expire_ts, context, score, emb) in _cache.items():
            if now > expire_ts:
                continue
            sim = float(q_emb @ emb)   # 均已归一化，点积 = 余弦
            if sim > best_sim:
                best_key, best_sim, best_val = k, sim, (context, score)

    if best_val is not None and best_sim >= config.RAG_CACHE_SEMANTIC_THRESHOLD:
        with _lock:
            _cache.move_to_end(best_key)   # LRU 更新
        _hits += 1
        return (best_val[0], best_val[1], "semantic")

    _misses += 1
    return None


def set_cached(query: str, context: str, score: float) -> None:
    if not config.RAG_CACHE_ENABLED:
        return
    key = normalize_query(query)
    if not key or not context:
        return

    emb = _get_embedder().encode(query, normalize_embeddings=True)
    with _lock:
        _cache.pop(key, None)
        _cache[key] = (time.time() + config.RAG_CACHE_TTL, context, score, emb)
        _cache.move_to_end(key)
        while len(_cache) > config.RAG_CACHE_MAX_SIZE:
            _cache.popitem(last=False)


def invalidate_cache() -> None:
    global _hits, _misses
    with _lock:
        _cache.clear()
        _hits = 0
        _misses = 0


def cache_stats() -> dict:
    with _lock:
        total = _hits + _misses
        return {
            "size": len(_cache),
            "hits": _hits,
            "misses": _misses,
            "hit_rate": round(_hits / total, 4) if total else 0.0,
        }
