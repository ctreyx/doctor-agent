"""LangGraph 图状态定义。

``State`` 是所有节点共享的数据结构，各节点通过返回值对字段做**增量更新**：

- ``messages`` 使用 ``add_messages`` reducer，多轮对话自动累积
  （LangGraph Server 依据 ``thread_id`` 自动 checkpoint）。
- 其余字段为普通覆盖写入（后写覆盖先写）。

字段语义详见各类型注解。新增字段时注意：若希望跨节点累加而非覆盖，
需为其指定 reducer（如 ``Annotated[list, operator.add]``）。
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    """医疗问答 Agent 的图状态。"""

    messages: Annotated[list[AnyMessage], add_messages]
    """对话历史，自动累积（user / assistant / feedback 消息均入此列）。"""

    context: str
    """检索参考资料（已按约定格式拼装，供生成节点注入 prompt）。"""

    score: float
    """重排后 top1 文档的相似度分数（用于检索 → 生成 / 联网的路由决策）。"""

    generation: str
    """当前生成的答案（供 Self-RAG 校验节点读取，也供测试脚本断言）。"""

    attempts: int
    """Self-RAG 校验循环计数，防止无限重生成。"""

    passed: bool
    """校验是否通过（幻觉校验 + 答题校验均通过为 True）。"""

    feedback: str
    """校验失败时产出的反馈，注入重生成节点指导修正。"""

    blocked: bool
    """是否被越权守卫拦截（True = 拒绝回答）。"""

    block_reply: str
    """被拦截时的合规回复话术。"""

    query: str
    """改写后的完整查询（消除多轮追问的指代漂移，供检索节点使用）。"""

    hyde_done: bool
    """是否已执行过 HyDE（防死循环，最多执行一次）。"""

    hyde_answer: str
    """HyDE 生成的假设答案（用于二次检索 + 可观测性追踪）。"""
