"""LangGraph 图组装。

把所有节点注册到 ``StateGraph`` 并按业务语义连线：

    START → guard ──blocked──→ blocked → END
              │
              └─(rewrite/retrieve)─→ retrieve ──score≥阈值──→ generate
                                                    │
                                          score<阈值─→ web_search → generate

    generate → grade ──不通过──→ feedback → generate（循环，上限 MAX_ATTEMPTS）
                    └─通过/达上限──→ END

对外接口：
    ``build_graph()`` 组装并编译图，返回 ``CompiledStateGraph``。
"""
from langgraph.graph import END, START, StateGraph

from doctor_agent.nodes.grade import feedback_node, grade_node, route_after_grade
from doctor_agent.nodes.generate import generate_node
from doctor_agent.nodes.guard import blocked_node, guard_node, route_after_guard
from doctor_agent.nodes.retrieve import retrieve_node, route
from doctor_agent.nodes.rewrite import rewrite_query_node
from doctor_agent.nodes.web_search import web_search_node
from doctor_agent.state import State


def build_graph():
    """组装并编译医疗问答 Agent 图。"""
    builder = StateGraph(State)

    # ---- 注册节点 ----
    builder.add_node("guard", guard_node)            # 越权拦截
    builder.add_node("blocked", blocked_node)        # 拒绝话术
    builder.add_node("rewrite", rewrite_query_node)  # 查询改写（消除语义漂移）
    builder.add_node("retrieve", retrieve_node)      # 混合检索 + 重排
    builder.add_node("web_search", web_search_node)  # 联网兜底
    builder.add_node("generate", generate_node)      # 生成答案
    builder.add_node("grade", grade_node)            # Self-RAG 校验
    builder.add_node("feedback", feedback_node)      # 反馈注入

    # ---- 连线 ----
    builder.add_edge(START, "guard")
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

    return builder.compile()
