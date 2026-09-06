

# doctor-agent —— 医疗问答 RAG + Agent

一个**生产级**的医疗问答 Agent：基于 **LangGraph** 编排的 RAG 流水线，融合了
混合检索、交叉编码重排、Self-RAG 自校验、越权拦截、多轮查询改写（消除语义漂移）等企业级能力，
并通过 **RAGAS** 做量化评测、用 **LLM 生成分析报告**。

> 一句话：用户提问 → 越权拦截 → 查询改写 → 混合检索/联网兜底 → 生成答案 → 自校验（不合格则重写），
> 全程可测试、可评测、可观测。


---

## 一、核心能力

| 能力 | 说明 |
|---|---|
| 混合检索 | Pinecone 向量检索 + BM25 关键词检索（`EnsembleRetriever`，权重 0.6/0.4） |
| 交叉编码重排 | `CrossEncoder(ms-marco-MiniLM-L-6-v2)` + Sigmoid 激活，把 logits 映射到 0~1 概率分 |
| 确定性路由 | rerank top1 分数 ≥ `RAG_MIN_SCORE`(0.6) 走本地 RAG，否则 Tavily 联网兜底 |
| Self-RAG 自校验 | 两个 grader：**幻觉校验**（答案是否基于检索事实）+ **答题校验**（是否解决用户问题），不合格带反馈重生成（最多 3 轮） |
| 越权守卫（Guard） | 在检索前拦截 6 类越权请求（开处方、伪造证明、自伤倾向、非医疗、色情暴力等） |
| 查询改写（Rewrite） | 多轮追问时把残句（"那会持续多久？"）改写成独立完整查询，消除**语义漂移** |
| RAGAS 评测 | 5 指标量化质量 + LLM 自动生成可交付的分析报告 |

---

## 二、技术栈

| 层 | 技术 |
|---|---|
| 语言/环境 | Python 3.13 + `uv` 包管理 |
| 编排 | LangGraph 1.2.11（`StateGraph` 状态机） |
| LLM | DeepSeek（`deepseek-chat`，经 `langchain-deepseek` 的 `init_chat_model`） |
| Embedding | 阿里云 DashScope `text-embedding-v3`（1024 维，自实现 `Embeddings` 接口） |
| 向量库 | Pinecone（索引 `demo`，1024 维，cosine，us-east-1） |
| 关键词检索 | BM25（`rank-bm25` / `langchain_community.BM25Retriever`） |
| 重排 | `sentence-transformers` CrossEncoder |
| 联网搜索 | Tavily Search |
| 评测 | RAGAS 0.2.12（5 指标 + LLM 分析报告） |

---

## 三、目录结构

```
doctor-agent/
├── main.py                    # 主程序（约 640 行）：数据加载 → 检索 → LangGraph 图
├── evaluate.py                # RAGAS 评测脚本（抽样本 → 检索生成 → 打分 → LLM 报告）
├── guard_eval.py              # 越权拦截测试（7 条用例，含 5 越权 + 2 正常对照）
├── test_semantic_drift.py     # 语义漂移对照实验（12 条追问用例，开/关 rewrite 对比）
├── datas/
│   └── zhongliu.csv           # 肿瘤科问答数据（GB18030 编码，4 列）
├── prompts/
│   └── doctor_prompt.md       # 医生角色 system prompt（含安全规则 + 免责声明模板）
├── pyproject.toml             # 依赖声明（uv）
├── langgraph.json             # LangGraph Server 配置（入口 ./main.py:graph）
├── cache/                     # 切分缓存（增量失效，避免重复切分）
├── src/doctor_agent/          # 包目录（目前基本为空，主逻辑在根目录 main.py）
├── .env                       # 密钥（不入库）
└── ragas_results.csv / ragas_report_*.md  # 评测输出
```

---

## 四、系统架构与处理流程

```mermaid
flowchart TD
    START((START)) --> G[guard<br/>越权拦截]
    G -->|blocked| B[blocked_node<br/>返回拒绝话术]
    B --> END1((END))

    G -->|正常| R[rewrite<br/>查询改写/消除指代]
    R --> RET[retrieve<br/>混合检索 + rerank]
    RET -->|score ≥ 0.6| GEN[generate<br/>注入参考资料生成]
    RET -->|score < 0.6| WEB[web_search<br/>Tavily 联网兜底]
    WEB --> GEN
    GEN --> GRD[grade<br/>Self-RAG 双校验]
    GRD -->|通过 或 达3次上限| END2((END))
    GRD -->|不通过| FB[feedback<br/>注入反馈]
    FB --> GEN
```

> `rewrite` 节点可用模块级开关 `ENABLE_REWRITE` 跳过（用于对照实验复现语义漂移）。

---

## 五、State 状态字段

定义在 `main.py` 的 `class State(TypedDict)`：

| 字段 | 类型 | 说明 |
|---|---|---|
| `messages` | `list[AnyMessage]`（`add_messages` reducer） | 对话历史，自动累积 |
| `context` | `str` | 检索参考资料（注入生成 prompt） |
| `score` | `float` | rerank top1 分数（用于路由） |
| `generation` | `str` | 当前生成的答案（供校验 + 测试读取） |
| `attempts` | `int` | 校验循环计数 |
| `passed` | `bool` | 校验是否通过 |
| `feedback` | `str` | 校验失败反馈（用于指导重生成） |
| `blocked` | `bool` | 是否被越权守卫拦截 |
| `block_reply` | `str` | 拦截时的合规回复话术 |
| `query` | `str` | 改写后的查询（解决语义漂移） |

---

## 六、各模块详解

### 1. 数据层（`main.py` 顶部）

- `clean_text()`：去 HTML 标签、全角空格、制表符，合并连续空白。
- `load_csv_documents()`：读 `datas/zhongliu.csv`（**GB18030 编码**），按行解析成 `Document`；短答案整行一条，长答案按 `RecursiveCharacterTextSplitter`（chunk 500 / overlap 80）切分，每块带上 `title + ask` 作为上下文。
- 缓存机制：`get_or_build_chunks()` 用 `PROCESS_VERSION + 原文` 的 MD5 做缓存 key；`cache/index_manifest.json` 记录每个 chunk 的 hash，只增量 upsert「新内容」。

### 2. Embedding 与向量库

- `DashScopeTextEmbeddingV3`：自实现 `Embeddings` 接口，调用 DashScope `text-embedding-v3`，1024 维，`query`/`document` 两种 text_type。
- Pinecone：索引 `demo`，增量写入由开关 `is_import_enabled` 控制（当前 `False`，跳过导入）。

### 3. 检索层

- `_get_retrieval()`：惰性单例，首次调用才初始化（CrossEncoder 下载 + BM25 建索引都很重，避免 Server 启动卡住）。
- `rerank_documents()`：CrossEncoder + Sigmoid 重排，返回 `[(score, doc)]` 降序，取 top 3。
- `route()`：`score ≥ 0.6` → `generate`；否则 → `web_search`。

### 4. 生成层

- `generate_node()`：把检索结果拼进 system prompt（`prompts/doctor_prompt.md` + `【参考资料】`），调 DeepSeek 生成。
- 历史上限 `MAX_MESSAGES = 20` 防超长。

### 5. Self-RAG 校验层

- `GradeHallucinations`：判断答案是否**基于检索事实**（防幻觉）。
- `GradeAnswer`：判断答案是否**解决了用户问题**（防答非所问）。
- 校验用低温模型 `temperature=0`。
- `route_after_grade()`：通过或 `attempts ≥ MAX_ATTEMPTS(3)` → 结束；否则 → `feedback` 注入反馈重生成。

### 6. 越权守卫（Guard）

- `GuardResult`：`is_blocked` / `reason` / `reply` 三个字段。
- `guard_prompt` 定义 6 类越权：非医疗、开处方/指定剂量、伪造证明、自伤/伤害意图、法律金融、色情暴力。
- `guard_node` 在检索前拦截；被拦截走 `blocked_node` 返回温和拒绝话术。

### 7. 查询改写（Rewrite，消除语义漂移）

- `rewrite_chain`：把「聊天历史 + 最新追问」改写成不依赖历史的完整查询。
- `rewrite_query_node`：取最近 6 条历史（3 轮），去掉最后一条追问，交给 LLM 改写；首轮无历史则原样返回。
- 开关 `ENABLE_REWRITE`：`True` 走 `guard → rewrite → retrieve`；`False` 直接 `guard → retrieve`（用于复现漂移）。

---

## 七、运行方式

### 1. 环境准备

`.env` 需要以下密钥：

```ini
DASHSCOPE_API_KEY=xxx   # 阿里云 DashScope（embedding）
PINECONE_API_KEY=xxx    # 向量库
DEEPSEEK_API_KEY=xxx    # LLM
TAVILY_API_KEY=xxx      # 联网搜索
# 可选：关掉追踪/镜像，减少噪音
LANGCHAIN_TRACING_V2=false
HF_ENDPOINT=https://hf-mirror.com
```

安装依赖：

```powershell
uv sync
```

### 2. 运行测试

```powershell
# 越权拦截测试（7 条用例）
& ".\.venv\Scripts\python.exe" guard_eval.py

# 语义漂移对照实验（12 条追问用例，开/关 rewrite 对比）
& ".\.venv\Scripts\python.exe" test_semantic_drift.py

# RAGAS 评测（抽 5 条样本，固定随机种子）
& ".\.venv\Scripts\python.exe" evaluate.py --n 5 --seed 42
```

### 3. 启动 LangGraph Server

```powershell
langgraph dev
```

入口由 `langgraph.json` 指定为 `./main.py:graph`。

---

## 八、评测（RAGAS）

`evaluate.py` 做五维量化评测：

| 指标 | 及格线 | 含义 |
|---|---|---|
| Faithfulness | 0.9 | 答案是否基于检索事实（防幻觉，医学场景最关键） |
| Answer Correctness | 0.8 | 答案是否正确 |
| Context Recall | 0.8 | 检索是否漏掉关键信息 |
| Context Precision | 0.7 | 检索是否带回无关内容 |
| Answer Relevancy | 0.7 | 答案是否紧扣问题 |

评测完成后：
1. 各指标均值打印到终端
2. 明细保存 `ragas_results.csv`
3. **LLM 自动生成分析报告**保存为 `ragas_report_YYYYMMDD_HHMMSS.md`（含逐指标诊断 + 最差样本剖析 + 修复建议）

---

## 九、关键参数速查

| 参数 | 位置 | 值 |
|---|---|---|
| `RAG_MIN_SCORE` | `main.py` | 0.6（低于此值走联网） |
| `MAX_ATTEMPTS` | `main.py` | 3（Self-RAG 重试上限） |
| `ENABLE_REWRITE` | `main.py` | True（语义漂移开关） |
| chunk_size / overlap | `main.py` | 500 / 80 |
| embedding 维度 | `main.py` | 1024（与 Pinecone 索引一致） |
| 检索 k / 重排 top_n | `main.py` | 10 / 3 |
| Ensemble 权重 | `main.py` | 向量 0.6 : BM25 0.4 |
| `MAX_MESSAGES` | `main.py` | 20（传给 LLM 的历史上限） |
| `PROCESS_VERSION` | `main.py` | "v1"（切分逻辑改动 +1 使缓存失效） |
| `is_import_enabled` | `main.py` | False（是否写向量库） |

---
