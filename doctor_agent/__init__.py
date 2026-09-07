"""doctor-agent 核心包。

一个生产级医疗问答 RAG + LangGraph Agent，模块职责划分如下：

- :mod:`config`          全局配置中心（路径、阈值、模型名等常量）
- :mod:`state`           LangGraph 图状态定义（State TypedDict）
- :mod:`document_loader` 医疗问答 CSV 的加载、清洗、切分与磁盘缓存
- :mod:`embeddings`      DashScope text-embedding-v3 封装
- :mod:`vector_store`    Pinecone 向量库初始化与增量写入
- :mod:`retrieval`       混合检索（向量 + BM25）与 CrossEncoder 重排
- :mod:`llm`             LLM 工厂（生成模型 / 低温校验模型）
- :mod:`prompts`         所有提示词模板
- :mod:`tools`           Agent 工具（本地检索 + Tavily 联网）
- :mod:`nodes`           图节点（guard / rewrite / retrieve / generate / grade 等）
- :mod:`graph`           图组装（build_graph）

对外入口在项目根目录 ``main.py``（LangGraph Server 通过 ``./main.py:graph`` 引用）。
"""

__version__ = "0.1.0"
