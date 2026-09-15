"""LangGraph 图组装。

把所有节点注册到 ``StateGraph`` 并按业务语义连线：

    START → guard ──blocked──→ blocked → END
              │
              └─(rewrite)─→ retrieve ──score≥阈值──→ generate
                                  │
                        score<阈值 且未 HyDE ──→ hyde ──→ retrieve（二次检索）
                                  │
                        score<阈值 且已 HyDE ──→ web_search → generate

    generate → grade ──不通过──→ feedback → generate（循环，上限 MAX_ATTEMPTS）
                    └─通过/达上限──→ END

对外接口：
    ``build_graph()`` 组装并编译图，返回 ``CompiledStateGraph``。
"""
from langgraph.graph import END, START, StateGraph

from doctor_agent.nodes.grade import feedback_node, grade_node, route_after_grade
from doctor_agent.nodes.generate import generate_node
from doctor_agent.nodes.guard import blocked_node, guard_node, route_after_guard
from doctor_agent.nodes.hyde import hyde_node
from doctor_agent.nodes.retrieve import retrieve_node, route
from doctor_agent.nodes.rewrite import rewrite_query_node
from doctor_agent.nodes.web_search import web_search_node
from doctor_agent.state import State



def route_entry(state: State) -> str:
    """起点条件边：续写走 generate 捷径，正常问题走 guard。

    续写能跳过检索，靠的是 checkpointer 已把上一轮的 ``context`` / ``query`` /
    ``score`` 恢复进 state，generate 直接读即可，重新检索纯属浪费。
    """
    return "generate" if state.get("resume") else "guard"


def route_after_generate(state: State) -> str:
    """生成后条件边：决定要不要跑 Self-RAG 校验。

    两类情况直接结束，不校验：

    - **被截断**（``truncated``）：截断的答案必然不完整，拿它去校验只会必然失败，
      白烧 2 次生成 + 2 次 grader；而且重生成的半截答案会污染 ``messages``。
      直接结束，交给用户点「继续生成」。
    - **续写**（``resume``）：续写内容是上一轮的延续，校验不通过会触发
      ``feedback → generate`` 把整段回答重写一遍，续写就白做了。
    """
    if state.get("truncated") or state.get("resume"):
        return "end"
    return "grade"


def build_graph(checkpointer=None):                    # ← 1
    """组装并编译医疗问答 Agent 图。"""
    builder = StateGraph(State)

    # ---- 注册节点 ----
    builder.add_node("guard", guard_node)            # 越权拦截
    builder.add_node("blocked", blocked_node)        # 拒绝话术
    builder.add_node("rewrite", rewrite_query_node)  # 查询改写（消除语义漂移）
    builder.add_node("retrieve", retrieve_node)      # 混合检索 + 重排
    builder.add_node("hyde", hyde_node)              # HyDE 假设答案二次检索
    builder.add_node("web_search", web_search_node)  # 联网兜底
    builder.add_node("generate", generate_node)      # 生成答案
    builder.add_node("grade", grade_node)            # Self-RAG 校验
    builder.add_node("feedback", feedback_node)      # 反馈注入

    # ---- 连线 ----
    builder.add_conditional_edges(
        START,
        route_entry,
        {"guard": "guard", "generate": "generate"},   #  续写直连 generate
    )
    builder.add_conditional_edges(
        "guard",
        route_after_guard,
        {"blocked": "blocked", "rewrite": "rewrite", "retrieve": "retrieve"},
    )
    builder.add_edge("blocked", END)          # 越权：直接结束，不走检索/生成
    builder.add_edge("rewrite", "retrieve")   # 改写后进检索

    builder.add_conditional_edges(
        "retrieve",
        route,
        {"generate": "generate", "web_search": "web_search", "hyde": "hyde"},
    )
    builder.add_edge("web_search", "generate")
    builder.add_edge("hyde", "retrieve")      # 假设答案 → 再检索

    # 生成后校验循环
    builder.add_conditional_edges(
        "generate",
        route_after_generate,
        {"grade": "grade", "end": END},               # ★ 续写生成完直接结束
    )
    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {"end": END, "regenerate": "feedback"},
    )
    builder.add_edge("feedback", "generate")

    return builder.compile(checkpointer=checkpointer)  # ← 2
