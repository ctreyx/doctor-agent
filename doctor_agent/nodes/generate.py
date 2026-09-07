"""生成节点。

把检索 / 联网得到的参考资料注入 system prompt，调用 LLM 生成答案，
并把答案写入 ``state["generation"]`` 供 Self-RAG 校验节点使用。
"""
from langchain_core.messages import SystemMessage

from doctor_agent import config
from doctor_agent.llm import get_llm
from doctor_agent.prompts import SYSTEM_PROMPT
from doctor_agent.state import State


def generate_node(state: State) -> dict:
    """节点：把参考资料注入 system prompt，调 LLM 生成答案。"""
    context = state.get("context") or "当前暂无相关参考资料。"
    sys_msg = SystemMessage(content=SYSTEM_PROMPT + "\n\n【参考资料】\n" + context)
    resp = get_llm().invoke([sys_msg] + state["messages"][-config.MAX_MESSAGES:])
    return {
        "messages": [resp],
        "generation": resp.content,   # 写回 state，供校验节点与测试读取
    }
