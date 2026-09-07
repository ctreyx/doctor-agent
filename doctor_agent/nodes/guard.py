"""越权守卫节点。

在检索前拦截越权请求（开处方、伪造证明、自伤倾向、非医疗、色情暴力等），
被拦截的请求直接走 ``blocked_node`` 返回温和拒绝话术，不进入检索 / 生成。

对外接口：
    ``guard_node``          越权判定节点；
    ``blocked_node``        拒绝话术节点；
    ``route_after_guard``   条件路由（blocked / rewrite / retrieve）。
"""
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from doctor_agent import config
from doctor_agent.llm import get_grader_llm
from doctor_agent.prompts import GUARD_PROMPT
from doctor_agent.state import State
from doctor_agent.tools import latest_user_query


class GuardResult(BaseModel):
    """越权判定结果（结构化输出）。"""

    is_blocked: bool = Field(description="是否应拒绝回答，true=拒绝")
    reason: str = Field(description="越权类型，用于日志审计")
    reply: str = Field(description="若拒绝，给用户的合规回复话术")


# 守卫链：低温模型 + 结构化输出，保证判定稳定
guard_chain = GUARD_PROMPT | get_grader_llm().with_structured_output(GuardResult)


def guard_node(state: State) -> dict:
    """节点：在检索前拦截越权请求。"""
    query = latest_user_query(state["messages"])
    r = guard_chain.invoke({"input": query})
    return {
        "blocked": r.is_blocked,
        "block_reply": r.reply if r.is_blocked else "",
    }


def blocked_node(state: State) -> dict:
    """节点：把拒绝话术作为最终回复返回。"""
    return {
        "messages": [
            AIMessage(content=state.get("block_reply") or "抱歉，这个问题我无法回答。")
        ]
    }


def route_after_guard(state: State) -> str:
    """条件边：越权 → 拦截；正常 → 改写（或跳过改写复现漂移）。

    读取 ``config.ENABLE_REWRITE``（模块属性访问，运行时可被测试脚本动态切换）。
    """
    if state.get("blocked"):
        return "blocked"
    return "rewrite" if config.ENABLE_REWRITE else "retrieve"
