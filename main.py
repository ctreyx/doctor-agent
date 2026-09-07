"""应用入口（薄入口）。

LangGraph Server 通过 ``langgraph.json`` 中的 ``./main.py:graph`` 引用本文件。

本文件只负责：
    1. 加载环境变量；
    2. 组装图（模块级单例 ``graph``）；
    3. 可选建库（受 ``config.IS_IMPORT_ENABLED`` 控制）；
    4. re-export 历史符号，兼容 ``evaluate.py`` / ``guard_eval.py`` /
       ``test_semantic_drift.py`` 中 ``import main as app`` 的访问方式。

所有业务逻辑均在 ``doctor_agent`` 包内。
"""
from dotenv import load_dotenv

# 1. 先加载 .env，确保 doctor_agent 各模块读取密钥时环境变量已就绪
load_dotenv()

from doctor_agent import config
from doctor_agent.graph import build_graph

# 2. 组装图（模块级单例，LangGraph Server 引用此变量，名称必须为 graph）
graph = build_graph()

# 3. 可选：首次建库时把 config.IS_IMPORT_ENABLED 置 True，增量写入向量库
if config.IS_IMPORT_ENABLED:
    from doctor_agent.document_loader import get_documents
    from doctor_agent.vector_store import upsert_documents
    upsert_documents(get_documents())

# ---------------------------------------------------------------------------
# 4. 向后兼容 re-export
#    历史脚本通过 `import main as app` 访问以下符号，此处保持接口不变。
#    - evaluate.py:         app.llm / app.embeddings / app.SYSTEM_PROMPT /
#                           app._get_retrieval / app.rerank_documents / app._format_context
#    - guard_eval.py:       app.graph
#    - test_semantic_drift: app.graph / app.rewrite_query_node / app.config.ENABLE_REWRITE
# ---------------------------------------------------------------------------
from doctor_agent.embeddings import get_embeddings
from doctor_agent.llm import get_llm
from doctor_agent.nodes.rewrite import rewrite_query_node
from doctor_agent.prompts import SYSTEM_PROMPT
from doctor_agent.retrieval import (
    format_context as _format_context,
    get_retrieval as _get_retrieval,
    rerank_documents,
)

# 惰性单例绑定为模块级属性（保持 app.llm / app.embeddings 直接可用；
# init_chat_model 与 embedding 封装构造均为轻量操作，不触发网络 IO）
llm = get_llm()
embeddings = get_embeddings()
