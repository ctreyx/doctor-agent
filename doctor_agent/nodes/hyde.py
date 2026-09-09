"""HyDE 节点：检索分值过低时，用假设答案二次检索。

HyDE（Hypothetical Document Embeddings）原理：
    用户查询往往很短、与知识库文档的 embedding 分布差异大，直接检索容易
    得分偏低。让 LLM 先为问题生成一段「假设的参考答案」，再用这段答案去
    检索——假设答案与真实文档在语义空间上更接近，能显著提升召回质量。

对外接口：
    ``hyde_node`` 生成假设答案并覆盖 ``state["query"]``，供检索节点复用。
"""
from langchain_core.output_parsers import StrOutputParser

from doctor_agent import config
from doctor_agent.llm import get_llm
from doctor_agent.prompts import HYDE_PROMPT
from doctor_agent.state import State
from doctor_agent.tools import latest_user_query

# HyDE 链：prompt → LLM → 字符串解析
hyde_chain = HYDE_PROMPT | get_llm() | StrOutputParser()


def hyde_node(state: State) -> dict:
    """节点：生成假设答案，并把其写入 ``query`` 供二次检索。

    容错设计（fail-open）：
        LLM 生成失败或产出为空时，不阻塞主流程——直接标记 ``hyde_done``，
        让路由自然落到联网兜底，保证用户始终能拿到答案。
    """
    question = state.get("query") or latest_user_query(state["messages"])
    original_score = state.get("score", 0.0)

    try:
        hyde_answer = hyde_chain.invoke({"question": question}).strip()
    except Exception as e:
        print(f"[hyde] 假设答案生成失败，回退联网：{e}")
        return {"hyde_done": True}

    if not hyde_answer:
        print("[hyde] 假设答案为空，回退联网")
        return {"hyde_done": True}

    print(f"[hyde] 原 score={original_score:.3f} < 阈值 {config.RAG_MIN_SCORE}，"
          f"已生成假设答案（{len(hyde_answer)} 字）重新检索")
    return {
        "query": hyde_answer,      # 覆盖 query，检索节点将用它二次检索
        "hyde_answer": hyde_answer,
        "hyde_done": True,
    }
