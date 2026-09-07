"""提示词集中管理。

所有 prompt 模板在此统一定义，便于版本化与调优。
``SYSTEM_PROMPT`` 读取外部 Markdown 文件（``prompts/doctor_prompt.md``），
业务逻辑与提示词内容解耦。
"""
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

from doctor_agent import config

# ---------------------------------------------------------------------------
# 医生角色 system prompt（外部文件，含安全规则与免责声明模板）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = Path(config.PROMPT_PATH).read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# 查询改写：消除多轮追问的指代漂移
# ---------------------------------------------------------------------------
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是查询改写器。给定聊天历史和最新的用户问题，
把问题改写成一个不依赖历史、可以独立理解的完整查询。
若问题本身已完整，原样返回。只输出改写后的查询，不要任何解释。"""),
    ("human", "聊天历史：\n{history}\n\n最新问题：{question}"),
])

# ---------------------------------------------------------------------------
# 越权守卫
# ---------------------------------------------------------------------------
GUARD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是医疗 AI 安全守卫。判断用户请求是否越权，必须拒绝。
越权范围：
1. 询问非医疗相关内容
2. 要求开具处方、指定药物剂量
3. 要求伪造医疗证明/病假条
4. 表达自伤、自杀或伤害他人意图
5. 法律、金融等非医疗领域专业建议
6. 色情、暴力、违法内容

越权：is_blocked=true，reply 写温和的拒绝话术（引导就医/求助，不激怒用户）。
正常医疗咨询：is_blocked=false。"""),
    ("human", "用户输入：{input}"),
])

# ---------------------------------------------------------------------------
# Self-RAG 校验（幻觉校验 + 答题校验）
# ---------------------------------------------------------------------------
HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是医学答案校验员。判断 AI 生成答案是否严格基于给定的参考资料，"
               "而不是凭空编造。给出 'yes' 或 'no'。"),
    ("human", "参考资料：\n\n {documents} \n\n AI 生成答案：{generation}"),
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是医学答案校验员。判断 AI 生成答案是否真正解决了用户的问题，"
               "没有答非所问或遗漏关键信息。给出 'yes' 或 'no'。"),
    ("human", "用户问题：\n\n {question} \n\n AI 生成答案：{generation}"),
])
