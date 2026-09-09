"""全局配置中心。

集中管理所有魔法数字、路径、模型名、阈值等常量，避免散落各模块难以维护。

生产环境演进建议：
    本文件为纯 Python 常量模块（零依赖、零副作用）。当需要按环境区分配置时，
    可平滑迁移到 ``pydantic-settings``，从环境变量 / YAML / 密钥管理服务读取，
    各业务模块的 ``from doctor_agent import config`` 接口保持不变。
"""
from pathlib import Path

# ============ 路径 ============
DATA_CSV_PATH = "./datas/zhongliu.csv"      # 医疗问答数据（GB18030 编码）
PROMPT_PATH = "prompts/doctor_prompt.md"    # 医生角色 system prompt
CACHE_DIR = Path("./cache")                 # 文档切分缓存目录
MANIFEST_PATH = CACHE_DIR / "index_manifest.json"   # 向量库增量写入清单

# ============ 文档切分 ============
CHUNK_SIZE = 500            # 长文本切分块大小（中文 500~800 常见）
CHUNK_OVERLAP = 80          # 块间重叠，建议 10%~20%
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ";", " "]
MAX_ANSWER_LEN = 800        # answer 超过该长度才触发切分
MIN_ANSWER_LEN = 5          # 过滤过短 / 垃圾行

# ============ Embedding ============
EMBEDDING_MODEL = "text-embedding-v3"   # 阿里云 DashScope
EMBEDDING_DIMENSION = 1024              # 必须与 Pinecone 索引维度一致
EMBEDDING_BATCH_SIZE = 10               # text-embedding-v3 单次最多 10 条

# ============ Pinecone 向量库 ============
PINECONE_INDEX_NAME = "demo"
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# ============ LLM ============
LLM_MODEL = "deepseek:deepseek-chat"    # 生成模型
GRADER_TEMPERATURE = 0.0                # 校验模型低温，保证判定稳定

# ============ 检索 ============
RETRIEVAL_K = 10                        # 向量 / BM25 各自召回条数
RERANK_TOP_N = 3                        # 重排后保留条数
RAG_MIN_SCORE = 0.6                     # rerank top1 分数阈值，低于则联网兜底
ENSEMBLE_WEIGHTS = [0.6, 0.4]           # 向量 : BM25 混合权重
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ============ Self-RAG 校验循环 ============
MAX_ATTEMPTS = 3        # 校验重试上限，防止死循环
MAX_MESSAGES = 20       # 传给 LLM 的历史消息上限，防止超长

# ============ Tavily 联网搜索 ============
TAVILY_MAX_RESULTS = 5

# ============ 缓存 ============
PROCESS_VERSION = "v1"  # 清洗/切分逻辑每次变更 +1，使旧缓存自动失效

# ============ 运行时开关 ============
ENABLE_REWRITE = True       # 语义漂移开关：True=走 rewrite（修复漂移）；False=跳过（复现漂移）
IS_IMPORT_ENABLED = False   # 是否将文档写入向量库（首次建库时置 True）


# ============ HyDE 检索增强 ============
HYDE_ENABLED = True   # 检索分数低时是否启用 HyDE 假设答案二次检索
