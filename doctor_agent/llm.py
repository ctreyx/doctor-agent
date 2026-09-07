"""LLM 工厂：生成模型与低温校验模型。

对外接口：
    ``get_llm()``        生成用模型（默认温度）；
    ``get_grader_llm()`` 校验用模型（``temperature=0``，保证判定稳定）。
"""
from langchain.chat_models import init_chat_model

from doctor_agent import config

_llm = None
_grader_llm = None


def get_llm():
    """返回生成模型的惰性单例。"""
    global _llm
    if _llm is None:
        _llm = init_chat_model(config.LLM_MODEL)
    return _llm


def get_grader_llm():
    """返回低温校验模型的惰性单例。

    Self-RAG 的幻觉校验 / 答题校验对稳定性要求高，
    温度置 0 可最大化判定确定性，避免随机性导致校验结果抖动。
    """
    global _grader_llm
    if _grader_llm is None:
        _grader_llm = init_chat_model(config.LLM_MODEL, temperature=config.GRADER_TEMPERATURE)
    return _grader_llm
