"""Self-RAG 校验节点。

用两个独立 grader 对生成答案做校验：

- **幻觉校验**（GradeHallucinations）：答案是否基于检索事实；
- **答题校验**（GradeAnswer）：答案是否解决了用户问题。

校验不通过时产出的 feedback 由 ``feedback_node`` 注入重生成；
``route_after_grade`` 控制循环上限（``config.MAX_ATTEMPTS``）防止死循环。
"""
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from doctor_agent import config
from doctor_agent.llm import get_grader_llm
from doctor_agent.prompts import ANSWER_PROMPT, HALLUCINATION_PROMPT
from doctor_agent.state import State
from doctor_agent.tools import latest_user_query


class GradeHallucinations(BaseModel):
    """判断答案是否基于检索事实，防幻觉。"""

    binary_score: str = Field(description="答案是否基于给定事实，'yes' 或 'no'")
    feedback: str = Field(description="若为 'no'，指出哪部分缺乏依据，用于指导修正")


class GradeAnswer(BaseModel):
    """判断答案是否解决了用户问题，防答非所问。"""

    binary_score: str = Field(description="答案是否解决了用户问题，'yes' 或 'no'")
    feedback: str = Field(description="若为 'no'，说明缺失了什么，用于指导修正")


# 两个校验链：低温模型 + 结构化输出
hallucination_grader = (
    HALLUCINATION_PROMPT | get_grader_llm().with_structured_output(GradeHallucinations)
)
answer_grader = (
    ANSWER_PROMPT | get_grader_llm().with_structured_output(GradeAnswer)
)


def grade_node(state: State) -> dict:
    """节点：对 generate 的答案做幻觉校验 + 答题校验。"""
    question = latest_user_query(state["messages"])
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
    msg = HumanMessage(
        content=f"【校验反馈】你的上一轮回答有问题：{fb}\n"
                f"请严格基于参考资料重新回答，确保准确且紧扣问题。"
    )
    return {"messages": [msg]}


def route_after_grade(state: State) -> str:
    """条件边：通过或达到上限 → 结束；否则 → 带反馈重新生成。"""
    if state.get("passed", False) or state.get("attempts", 0) >= config.MAX_ATTEMPTS:
        return "end"
    return "regenerate"
