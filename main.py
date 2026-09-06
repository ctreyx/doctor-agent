import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_pinecone import PineconeVectorStore
from sentence_transformers import CrossEncoder


from torch import nn                                   # sigmoid 用
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model

load_dotenv()

# 1. 读取文档 CSV 按行解析成带 metadata 的 Document
import csv
from langchain_core.documents import Document


# 表格行数据：不切或按记录切（上面已经一行一条，直接可用）
# 长文本（如病历描述字段）：才用递归切割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 中文长文本 500~800 常见
    chunk_overlap=80,      # 建议 10%~20%，你原来 50/500=10% 偏小
    separators=["\n\n", "\n", "。", "！", "？", ";", " "],
)



# 清晰逻辑：对文本进行清理，去除 HTML 标签、全角空格、制表符，并将连续空白合并为单个空格
import re
import html as html_mod

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = html_mod.unescape(s)                     # &amp; → &
    s = re.sub(r"<[^>]+>", "", s)                # 去 HTML 标签
    s = s.replace("\u3000", " ").replace("\t", " ")  # 全角空格/制表符 → 空格
    s = re.sub(r"\s+", " ", s)                   # 连续空白 → 单个空格
    return s.strip()

def load_csv_documents(path: str, max_len: int = 800) -> list[Document]:
    docs = []
    seen = set()          # 去重
    splitter = text_splitter
    with open(path, "r", encoding="gb18030") as f:
         for i, row in enumerate(csv.DictReader(f)):

             # --- 清洗 ---
            title  = clean_text(row.get("title")  or "")
            ask    = clean_text(row.get("ask")    or "")
            answer = clean_text(row.get("answer") or "")
            if len(answer) < 5:                 # 过滤过短/垃圾行
                continue
            key = f"{title}|{ask}"
            if key in seen:                     # 去重
                continue
            seen.add(key)
            row = {"department": row.get("department"), "title": title,
                   "ask": ask, "answer": answer}
            # --- 清洗结束，下面是原来的逻辑 ---



            base_meta = {"source": path, "row_id": i, "department": row.get("department")}

            if len(row.get("answer") or "") <= max_len:
                # 正常记录：一行一条，不切
                text = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
                docs.append(Document(page_content=text, metadata={**base_meta, **row}))
            else:
                # 超长记录：只切 answer，每块都带上 title+ask 作为上下文
                head = f"title: {row.get('title')}\nask: {row.get('ask')}"
                for j, piece in enumerate(splitter.split_text(row["answer"])):
                    docs.append(Document(
                        page_content=f"{head}\nanswer: {piece}",
                        metadata={**base_meta, "title": row.get("title"),
                                  "chunk_id": j},
                    ))
    return docs


import hashlib, json, os
from pathlib import Path

CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

PROCESS_VERSION = "v1"   # 清洗/切分逻辑每改一次就 +1，缓存自动失效

def content_hash(text: str) -> str:
    return hashlib.md5((PROCESS_VERSION + text).encode("utf-8")).hexdigest()

MANIFEST_PATH = CACHE_DIR / "index_manifest.json"

def doc_hash(d: Document) -> str:
    payload = d.page_content + json.dumps(d.metadata, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}

def save_manifest(m: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")

def get_or_build_chunks(path: str) -> list[Document]:
    raw = Path(path).read_text(encoding="gb18030")
    h = content_hash(raw)
    cache_file = CACHE_DIR / f"{Path(path).stem}.{h}.json"

    if cache_file.exists():              # 命中缓存：直接读，不重新切
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return [Document(**d) for d in data]

    docs = load_csv_documents(path)      # 没命中：才切一次
    cache_file.write_text(
        json.dumps([{"page_content": d.page_content, "metadata": d.metadata}
                    for d in docs], ensure_ascii=False),
        encoding="utf-8",
    )
    return docs


docs = get_or_build_chunks("./datas/zhongliu.csv")


# print(f"Loaded {len(docs)} documents from CSV, sample:\n{docs[0].page_content}\n{docs[0].metadata}")


#  之前都是rag切片逻辑

# #  现在开始连接数据库
# 4) Embedding：自实现，基于官方 dashscope SDK
#    （langchain-dashscope / langchain-community 已停更，与 LangChain 1.x 不兼容，故弃用）
from typing import List
from dashscope import TextEmbedding
from langchain_core.embeddings import Embeddings
from pinecone import Pinecone, ServerlessSpec


class DashScopeTextEmbeddingV3(Embeddings):
    """阿里云 text-embedding-v3，实现 LangChain Embeddings 接口"""

    def __init__(self, api_key: str, model: str = "text-embedding-v3",
                 dimension: int = 1024, batch_size: int = 10):
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size   # text-embedding-v3 单次最多 10 条

    def _embed(self, texts: List[str], text_type: str) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            resp = TextEmbedding.call(
                model=self.model,
                input=texts[i:i + self.batch_size],
                dimension=self.dimension,
                text_type=text_type,   # query / document
                api_key=self.api_key,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"embedding 失败: code={resp.code} msg={resp.message}")
            out.extend(item["embedding"] for item in resp.output["embeddings"])
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts, text_type="document")

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], text_type="query")[0]


embeddings = DashScopeTextEmbeddingV3(
    api_key=os.getenv("DASHSCOPE_API_KEY", ""),
    dimension=1024,   # 必须与 Pinecone index 的 dimension 一致
)

print(f"Loaded {embeddings} embeddings from DashScope")

# 1) 建客户端
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))

# 2) 建/连索引：维度必须和 embedding 一致
index_name = "demo"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,          # text-embedding-v3 默认 1024 维
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
index = pc.Index(index_name)

# 3) 写入：必须传稳定 ids，避免重复写入
from langchain_pinecone import PineconeVectorStore


ids = [
    f"zhongliu-{d.metadata['row_id']}"
    + (f"-{d.metadata['chunk_id']}" if "chunk_id" in d.metadata else "")
    for d in docs
]

# 只处理「新 id」或「内容变了」的 chunk
manifest = load_manifest()
to_upsert = [(i, d, doc_hash(d)) for i, d in zip(ids, docs)
             if manifest.get(i) != doc_hash(d)]

print(f"总 {len(docs)} 条，本次增量 {len(to_upsert)} 条")

# 是否开启向量库导入
is_import_enabled = False   # 设置为 False 可关闭向量库导入

if to_upsert and is_import_enabled:
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
    BATCH = 100
    for start in range(0, len(to_upsert), BATCH):
        batch = to_upsert[start:start + BATCH]
        vectorstore.add_documents(
            documents=[d for _, d, _ in batch],
            ids=[i for i, _, _ in batch],
            batch_size=BATCH,
        )
        for i, _, h in batch:
            manifest[i] = h
        save_manifest(manifest)   # 每批落盘，断点续跑

print(index.describe_index_stats())



# 向量库导入完成


# 6) BM25 检索器
from langchain_community.retrievers import BM25Retriever   # ✅ 官方推荐
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder


# 检索资源惰性单例：CrossEncoder 下载 + BM25 建索引都很重，
# 首次调用 search_medical_knowledge 时才初始化，避免 Server 启动时卡住
_retrieval = None

def _get_retrieval():
    global _retrieval
    if _retrieval is None:
        # 初始化 CrossEncoder reranker
        reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            activation_fn=nn.Sigmoid(),   # logits(-10~10) → 概率(0~1)
        )

        vs = PineconeVectorStore(index=index, embedding=embeddings)
        vector_retriever = vs.as_retriever(search_kwargs={"k": 10})
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 10
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.6, 0.4],
        )
        _retrieval = (ensemble, reranker)
    return _retrieval


# 9) CrossEncoder 重排
def rerank_documents(query: str, docs: list[Document], top_n: int = 3):
    if not docs:                       # 空结果兜底
        return []
    _, reranker = _get_retrieval()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return ranked[:top_n]


# 10) 把 RAG 检索包装成工具，供 agent 调用（Agentic RAG）
from langchain_core.tools import tool


def _format_context(docs: list[Document]) -> str:
    """按 system_prompt 约定的格式组织检索结果。"""
    if not docs:
        return "当前暂无相关参考资料。"
    lines = ["【检索结果开始】"]
    for i, doc in enumerate(docs, start=1):
        dept = doc.metadata.get("department", "未知科室")
        title = doc.metadata.get("title", "")
        lines.append(f"[{i}] 来源：{dept} | 标题：{title} | 内容：{doc.page_content}")
    lines.append("【检索结果结束】")
    return "\n".join(lines)

RAG_MIN_SCORE = 0.6


def _format_web_result(raw: str) -> str:
    """给 tavily 结果包一层来源标注。"""
    return "【联网搜索结果开始】\n" + raw + "\n【联网搜索结果结束】"


def _latest_user_query(messages: list) -> str:
    """取最近一条用户消息（跳过 AI/工具消息，企业级健壮性）。"""
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
    ensemble, _ = _get_retrieval()
    candidates = ensemble.invoke(query)                  # 向量+BM25 混合检索
    top = rerank_documents(query, candidates, top_n=3)   # CrossEncoder 重排
    return _format_context([d for _, d in top])



# 创建agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents import create_agent
from langchain_tavily import TavilySearch

tavily_tool = TavilySearch(
    max_results=5,
    topic="general",
)


# 注意：LangGraph Server 模式下持久化由平台自动管理（thread_id 即会话标识），
# 图定义里不能带自定义 checkpointer，所以这里不再手动配置 SqliteSaver。

middleware = [
  SummarizationMiddleware(
    model="deepseek:deepseek-chat",
    trigger=("tokens", 4000),
    keep=("messages", 20)
  )
]



SYSTEM_PROMPT = Path("prompts/doctor_prompt.md").read_text(encoding="utf-8")

# graph = create_agent(
#     model="deepseek:deepseek-chat",
#     system_prompt=SYSTEM_PROMPT,
#     tools=[tavily_tool, search_medical_knowledge],
#     middleware=middleware,
# )



# State 用 add_messages reducer：多轮对话自动累积，LangGraph Server 自动 checkpoint
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    context: str        # 检索参考资料
    score: float        # rerank top1 分数
    generation: str     # 当前生成的答案（供校验用）
    attempts: int       # 校验循环计数
    passed: bool        # 校验是否通过
    feedback: str       # 校验失败时的反馈，用于指导重生成

    blocked: bool        # 判断是否越级：是否被守卫拦截
    block_reply: str     # 判断是否越级：拦截时的合规回复话术

    query: str          # 解决语义飘逸问题


llm = init_chat_model("deepseek:deepseek-chat")
MAX_MESSAGES = 20       # 传给 LLM 的历史消息上限，防超长


def retrieve_node(state: State) -> dict:
    """节点 1：混合检索 + rerank，把结果和分数写回 state。"""
    # query = _latest_user_query(state["messages"])
    query = state.get("query") or _latest_user_query(state["messages"])   # 解决语义飘逸问题
    ensemble, _ = _get_retrieval()
    ranked = rerank_documents(query, ensemble.invoke(query), top_n=3)
    score = ranked[0][0] if ranked else 0.0
    return {
        "context": _format_context([d for _, d in ranked]),
        "score": score,
        "query": query,   # ← 新增：把实际用的 query 写回 state
    }


def route(state: State) -> str:
    """条件边：分数达标 → 直接生成；否则 → 联网。"""
    return "generate" if state.get("score", 0.0) >= RAG_MIN_SCORE else "web_search"


def web_search_node(state: State) -> dict:
    """节点 2：RAG 不够时，tavily 联网兜底。"""
    # query = _latest_user_query(state["messages"])

    query = state.get("query") or _latest_user_query(state["messages"])  # 解决语义飘逸问题
    try:
        raw = tavily_tool.invoke({"query": query})
        return {"context": _format_web_result(raw)}
    except Exception:
        # 联网失败 → 退回本地结果
        return {"context": state.get("context", "当前暂无相关参考资料。")}


def generate_node(state: State) -> dict:
    """节点 3：把参考资料注入 system prompt，调 LLM 生成答案。"""
    context = state.get("context") or "当前暂无相关参考资料。"
    sys_msg = SystemMessage(
        content=SYSTEM_PROMPT + "\n\n【参考资料】\n" + context
    )
    resp = llm.invoke([sys_msg] + state["messages"][-MAX_MESSAGES:])
    return {
        "messages": [resp],
        "generation": resp.content,   # ← 新增：把答案写进 state 供校验
    }


# ============ Self-RAG 校验层：两个独立 grader（官方 langgraph_self_rag 标准） ============
grader_llm = init_chat_model("deepseek:deepseek-chat", temperature=0)  # 校验用低温模型

class GradeHallucinations(BaseModel):
    """判断答案是否基于检索事实，防幻觉。"""
    binary_score: str = Field(description="答案是否基于给定事实，'yes' 或 'no'")
    feedback: str = Field(description="若为 'no'，指出哪部分缺乏依据，用于指导修正")

class GradeAnswer(BaseModel):
    """判断答案是否解决了用户问题，防答非所问。"""
    binary_score: str = Field(description="答案是否解决了用户问题，'yes' 或 'no'")
    feedback: str = Field(description="若为 'no'，说明缺失了什么，用于指导修正")

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是医学答案校验员。判断 AI 生成答案是否严格基于给定的参考资料，"
               "而不是凭空编造。给出 'yes' 或 'no'。"),
    ("human", "参考资料：\n\n {documents} \n\n AI 生成答案：{generation}"),
])
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是医学答案校验员。判断 AI 生成答案是否真正解决了用户的问题，"
               "没有答非所问或遗漏关键信息。给出 'yes' 或 'no'。"),
    ("human", "用户问题：\n\n {question} \n\n AI 生成答案：{generation}"),
])

hallucination_grader = hallucination_prompt | grader_llm.with_structured_output(GradeHallucinations)
answer_grader = answer_prompt | grader_llm.with_structured_output(GradeAnswer)






# 越权
class GuardResult(BaseModel):
    """越权判定结果。"""
    is_blocked: bool = Field(description="是否应拒绝回答，true=拒绝")
    reason: str = Field(description="越权类型，用于日志审计")
    reply: str = Field(description="若拒绝，给用户的合规回复话术")

guard_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是医疗 AI 安全守卫。判断用户请求是否越权，必须拒绝。
越权范围：
1. 询问非医疗相关内容
2. 要求开具处方、指定药物剂量
3. 要求伪造医疗证明/病假条
4. 表达自伤、自杀或伤害他人意图
5. 法律、金融等非医疗领域专业建议
6. 色情、暴力、违法内容

越权：is_blocked=true，reply 写温和的拒绝话术（引导就医/求助，不激怒用户）。
正常医疗咨询：is_blocked=false。"""),
    ("human", "用户输入：{input}"),
])

guard_chain = guard_prompt | grader_llm.with_structured_output(GuardResult)

def guard_node(state: State) -> dict:
    """节点：在检索前拦截越权请求。"""
    query = _latest_user_query(state["messages"])
    r = guard_chain.invoke({"input": query})
    return {
        "blocked": r.is_blocked,
        "block_reply": r.reply if r.is_blocked else "",
    }


ENABLE_REWRITE = True   # 语义漂移开关：True=走 rewrite（修复漂移）；False=跳过（复现漂移）
# def route_after_guard(state: State) -> str:
#     """条件边：越权 → 直接拦截结束；正常 → 进检索。"""
#     return "blocked" if state.get("blocked") else "retrieve"
#     # return "blocked" if state.get("blocked") else "rewrite"

def route_after_guard(state: State) -> str:
    """条件边：越权 → 拦截；正常 → 改写（或跳过改写复现漂移）。"""
    if state.get("blocked"):
        return "blocked"
    return "rewrite" if ENABLE_REWRITE else "retrieve"



def blocked_node(state: State) -> dict:
    """节点：把拒绝话术作为最终回复返回。"""
    return {"messages": [AIMessage(content=state.get("block_reply") or "抱歉，这个问题我无法回答。")]}

MAX_ATTEMPTS = 3   # 最多校验 3 轮，防死循环（官方教程没加，企业级必须加）

def grade_node(state: State) -> dict:
    """节点：对 generate 的答案做幻觉校验 + 答题校验。"""
    question = _latest_user_query(state["messages"])
    context = state.get("context", "")
    generation = state.get("generation", "")
    attempts = state.get("attempts", 0) + 1

    try:
        h = hallucination_grader.invoke({"documents": context, "generation": generation})
        if h.binary_score == "no":
            return {"attempts": attempts, "passed": False, "feedback": h.feedback}

        a = answer_grader.invoke({"question": question, "generation": generation})
        if a.binary_score == "no":
            return {"attempts": attempts, "passed": False, "feedback": a.feedback}

        return {"attempts": attempts, "passed": True, "feedback": ""}
    except Exception as e:
        # grader 技术性故障 → fail-open 放行，避免用户永远拿不到答案
        print(f"[grade] 校验器异常，放行：{e}")
        return {"attempts": attempts, "passed": True, "feedback": ""}


def feedback_node(state: State) -> dict:
    """节点：把校验反馈追加为一条消息，指导 LLM 修正上一轮答案。"""
    fb = state.get("feedback") or "上一轮回答不合格，请重新作答。"
    msg = HumanMessage(content=f"【校验反馈】你的上一轮回答有问题：{fb}\n"
                               f"请严格基于参考资料重新回答，确保准确且紧扣问题。")
    return {"messages": [msg]}


def route_after_grade(state: State) -> str:
    """条件边：通过或达到上限 → 结束；否则 → 带反馈重新生成。"""
    if state.get("passed", False) or state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "end"
    # if state.get("attempts", 0) >= MAX_ATTEMPTS:
    #     print(f"[grade] ⚠️ 达到最大重试次数 {MAX_ATTEMPTS}，强制放行（答案质量可能不佳）")
    #     # 可选：在 state 中标记，让前端展示"此回答未通过完整校验"的提示
    #     return "end"
    return "regenerate"



# 查询改写：消除多轮追问的指代漂移
from langchain_core.output_parsers import StrOutputParser
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是查询改写器。给定聊天历史和最新的用户问题，
把问题改写成一个不依赖历史、可以独立理解的完整查询。
若问题本身已完整，原样返回。只输出改写后的查询，不要任何解释。"""),
    ("human", "聊天历史：\n{history}\n\n最新问题：{question}"),
])

rewrite_chain = rewrite_prompt | llm | StrOutputParser()


def rewrite_query_node(state: State) -> dict:
    """节点：把追问句改写为独立查询，消除指代漂移。"""
    msgs = state["messages"]
    question = _latest_user_query(msgs)   # 当前问题
    history = msgs[:-1]                   # 去掉最后一条（即当前问题）

    # 首轮对话没有历史，跳过改写，省一次 LLM 调用
    if not history:
        return {"query": question}

    history_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in history[-6:]             # 最多取最近 3 轮对话
    )
    new_q = rewrite_chain.invoke({"history": history_text, "question": question})
    return {"query": new_q.strip()}



builder = StateGraph(State)
builder.add_node("retrieve", retrieve_node)
builder.add_node("web_search", web_search_node)
builder.add_node("generate", generate_node)
builder.add_node("grade", grade_node)          # 校验
builder.add_node("feedback", feedback_node)    # 反馈注入

#  越级判断
builder.add_node("guard", guard_node)
builder.add_node("blocked", blocked_node)
builder.add_node("rewrite", rewrite_query_node)     # ← 新增 语义飘逸


builder.add_edge(START, "guard")
builder.add_conditional_edges(
    "guard",
    route_after_guard,
    #   {"blocked": "blocked", "rewrite": "rewrite"},
    {"blocked": "blocked", "rewrite": "rewrite", "retrieve": "retrieve"},

)
builder.add_edge("blocked", END)          # 越权：直接结束，不走检索/生成
builder.add_edge("rewrite", "retrieve")     # ← 新增语义飘逸：改写后进检索

builder.add_conditional_edges(
    "retrieve",
    route,
    {"generate": "generate", "web_search": "web_search"},
)
builder.add_edge("web_search", "generate")

# 生成后校验循环
builder.add_edge("generate", "grade")
builder.add_conditional_edges(
    "grade",
    route_after_grade,
    {"end": END, "regenerate": "feedback"},
)
builder.add_edge("feedback", "generate")

graph = builder.compile()   # 变量名必须叫 graph



# if __name__ == "__main__":
#     question = "食道癌手术后如何恢复？"
#     config = {"configurable": {"thread_id": "session-001"}}
#     result = graph.invoke(
#         {"messages": [{"role": "user", "content": question}]},
#         config=config,
#     )
#     print(result["messages"][-1].content)
