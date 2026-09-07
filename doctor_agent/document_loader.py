"""文档加载与缓存。

职责：
1. 从 GB18030 编码的 CSV 中解析医疗问答数据为 LangChain ``Document``；
2. 文本清洗（HTML 标签、全角空格、制表符、连续空白）；
3. 长答案递归切分（短答案整条保留）；
4. 磁盘缓存 + 内容 hash 增量失效，避免重复切分。

对外接口：
    ``get_documents()`` —— 惰性单例，首次调用时加载（优先命中缓存）。
"""
import csv
import hashlib
import html as html_mod
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from doctor_agent import config

# ---------------------------------------------------------------------------
# 切分器：模块级单例（配置不变，无需重复构造）
# ---------------------------------------------------------------------------
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    separators=config.CHUNK_SEPARATORS,
)


def clean_text(s: str) -> str:
    """清洗文本：HTML 反转义 → 去标签 → 全角空格/制表符 → 合并空白 → strip。"""
    if not s:
        return ""
    s = html_mod.unescape(s)                         # &amp; → &
    s = re.sub(r"<[^>]+>", "", s)                    # 去 HTML 标签
    s = s.replace("\u3000", " ").replace("\t", " ")  # 全角空格 / 制表符 → 空格
    s = re.sub(r"\s+", " ", s)                       # 连续空白 → 单个空格
    return s.strip()


def load_csv_documents(path: str, max_len: int = config.MAX_ANSWER_LEN) -> list[Document]:
    """读取 CSV 并解析为 Document 列表。

    - 过滤：``answer`` 长度 < ``config.MIN_ANSWER_LEN`` 的垃圾行；
    - 去重：以 ``title|ask`` 为键；
    - 短答案：整行一条 Document；
    - 长答案：只切 ``answer``，每块携带 ``title + ask`` 头部作为上下文，
      并在 metadata 中记录 ``chunk_id``。
    """
    docs: list[Document] = []
    seen: set[str] = set()

    with open(path, "r", encoding="gb18030") as f:
        for i, raw in enumerate(csv.DictReader(f)):
            # --- 清洗字段 ---
            title = clean_text(raw.get("title") or "")
            ask = clean_text(raw.get("ask") or "")
            answer = clean_text(raw.get("answer") or "")

            if len(answer) < config.MIN_ANSWER_LEN:   # 过滤过短 / 垃圾行
                continue
            key = f"{title}|{ask}"
            if key in seen:                           # 去重
                continue
            seen.add(key)

            row = {
                "department": raw.get("department"),
                "title": title,
                "ask": ask,
                "answer": answer,
            }

            base_meta = {
                "source": path,
                "row_id": i,
                "department": raw.get("department"),
            }

            if len(answer) <= max_len:
                # 正常记录：一行一条，不切
                text = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
                docs.append(Document(page_content=text, metadata={**base_meta, **row}))
            else:
                # 超长记录：只切 answer，每块带上 title+ask 作为上下文
                head = f"title: {row['title']}\nask: {row['ask']}"
                for j, piece in enumerate(_text_splitter.split_text(row["answer"])):
                    docs.append(Document(
                        page_content=f"{head}\nanswer: {piece}",
                        metadata={**base_meta, "title": row["title"], "chunk_id": j},
                    ))
    return docs


# ---------------------------------------------------------------------------
# 缓存：内容 hash 变化才重新切分，清单（manifest）用于向量库增量写入
# ---------------------------------------------------------------------------
def content_hash(text: str) -> str:
    """原文内容的稳定指纹（PROCESS_VERSION 参与，保证清洗逻辑变更后缓存失效）。"""
    return hashlib.md5((config.PROCESS_VERSION + text).encode("utf-8")).hexdigest()


def doc_hash(d: Document) -> str:
    """单个 Document 的内容指纹（正文 + 元数据），用于判断是否需要重新 upsert。"""
    payload = d.page_content + json.dumps(d.metadata, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def load_manifest() -> dict:
    """读取向量库增量写入清单（id → 内容 hash）。"""
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(m: dict) -> None:
    """落盘向量库增量写入清单。"""
    config.CACHE_DIR.mkdir(exist_ok=True)
    config.MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")


def _get_or_build_chunks(path: str) -> list[Document]:
    """优先命中磁盘缓存，未命中才切分并写缓存。"""
    raw = Path(path).read_text(encoding="gb18030")
    h = content_hash(raw)
    cache_file = config.CACHE_DIR / f"{Path(path).stem}.{h}.json"

    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return [Document(**d) for d in data]

    docs = load_csv_documents(path)
    config.CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(
        json.dumps(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return docs


# ---------------------------------------------------------------------------
# 惰性单例：文档只加载一次
# ---------------------------------------------------------------------------
_docs: list[Document] | None = None


def get_documents(path: str = config.DATA_CSV_PATH) -> list[Document]:
    """返回全量文档的惰性单例。

    首次调用执行加载（读缓存或切分），后续调用直接复用；
    避免模块 import 阶段就触发 IO 与切分，缩短 Server 启动时间。
    """
    global _docs
    if _docs is None:
        _docs = _get_or_build_chunks(path)
    return _docs
