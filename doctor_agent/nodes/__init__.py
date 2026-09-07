"""LangGraph 图节点集合。

各节点职责单一、彼此解耦，由 :mod:`doctor_agent.graph` 负责组装与连线。
节点通过 ``State`` 交换数据，不直接互相调用。
"""
