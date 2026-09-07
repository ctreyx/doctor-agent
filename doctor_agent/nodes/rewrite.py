"""查询改写节点。

把多轮对话中的追问残句（如"那会持续多久？"）改写成不依赖历史的完整查询
（如"化疗后恶心会持续多久？"），消除检索时的语义漂移。

对外接口：
    ``rewrite_query_node``  改写节点。
"""
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from doctor_agent.llm import get_llm
from doctor_agent.prompts import REWRITE_PROMPT
from doctor_agent.state import State
from doctor_agent.tools import latest_user_query

# 改写链：prompt → LLM → 字符串解析
rewrite_chain = REWRITE_PROMPT | get_llm() | StrOutputParser()


def rewrite_query_node(state: State) -> dict:
    """节点：把追问句改写为独立查询，消除指代漂移。"""
    msgs = state["messages"]
    question = latest_user_query(msgs)   # 当前问题
    history = msgs[:-1]                  # 去掉最后一条（即当前问题）

    # 首轮对话没有历史，跳过改写，省一次 LLM 调用
    if not history:
        return {"query": question}

    history_text = "\n".join(
        f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
        for m in history[-6:]            # 最多取最近 3 轮对话
    )
    new_q = rewrite_chain.invoke({"history": history_text, "question": question})
    return {"query": new_q.strip()}
