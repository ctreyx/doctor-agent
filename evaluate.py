"""
RAGAS 评测脚本：量化评测 doctor-agent 的 RAG 检索与生成质量。

一句话理解它干什么：
    1. 从 CSV 里随机抽 N 条问答（问题 + 参考答案）
    2. 对每条：走一遍你的「检索 → 生成」流程，拿到检索结果和 AI 答案
    3. 把 (问题, 检索结果, AI答案, 参考答案) 四样东西喂给 ragas 打分
    4. 输出各项指标均值 + 保存明细 CSV

运行方式（在项目根目录执行）：
    .venv\\Scripts\\python.exe evaluate.py --n 5 --seed 42
"""
import csv          # 读 CSV 数据文件
import random       # 随机抽样
import argparse     # 解析命令行参数 --n / --seed
import json         # 序列化评测结果给 LLM
from datetime import datetime   # 报告文件名时间戳
from dotenv import load_dotenv   # 读取 .env 文件里的 API key

load_dotenv()       # 先把环境变量加载进来（DeepSeek / DashScope / Pinecone 的 key）

# ================= 复用主工程 =================
# import main 会执行 main.py 顶部的初始化逻辑：
#   读文档缓存、连接 Pinecone、构建检索器、编译 LangGraph 图（约十几秒）
# 这样评测用的就是「线上同一套」检索器和 LLM，而不是另写一份，
# 否则评测结果不能反映线上真实质量。
import main as app

from langchain_core.messages import SystemMessage, HumanMessage   # 构造发给 LLM 的消息


def load_samples(n: int, seed: int) -> list[dict]:
    """
    从 CSV 里随机抽 n 条样本。

    每条样本只有两个字段：
      - question     : 用户问题（取自 CSV 的 ask 列）
      - ground_truth : 参考答案（取自 CSV 的 answer 列，用来对照 AI 答案是否答对）

    参数 n    : 抽多少条（建议先 5 条验证能跑通，再放大到 50+）
    参数 seed : 随机种子，固定后每次抽到同样样本，保证结果可复现
    """
    random.seed(seed)
    rows = []
    with open("./datas/zhongliu.csv", "r", encoding="gb18030") as f:
        for row in csv.DictReader(f):
            ask = (row.get("ask") or "").strip()
            answer = (row.get("answer") or "").strip()
            # 过滤太短的脏数据：问题至少 5 字，答案至少 10 字
            if len(ask) >= 5 and len(answer) >= 10:
                rows.append({"question": ask, "ground_truth": answer})
    return random.sample(rows, n)


def run_pipeline(samples: list[dict]) -> list[dict]:
    """
    核心函数：对每条样本走一遍「检索 → 生成」，产出 ragas 需要的记录。

    ragas 0.2.x 的四要素（注意是【新字段名】）：
      - user_input         : 用户问题
      - retrieved_contexts : 检索到的上下文列表（list[str]）
      - response           : AI 生成的答案
      - reference          : 参考答案（对照标准）

    返回：list[dict]，每个 dict 是一条评测样本（记录式，不是列式）
    """
    ensemble, _ = app._get_retrieval()   # 拿到混合检索器（Pinecone 向量 + BM25）

    records = []
    for i, s in enumerate(samples, 1):
        q = s["question"]

        # ① 检索：混合检索 + CrossEncoder 重排，取 top 3
        ranked = app.rerank_documents(q, ensemble.invoke(q), top_n=3)
        contexts = [d.page_content for _, d in ranked]          # 只取正文
        ctx_str = app._format_context([d for _, d in ranked])   # 格式化成给 LLM 的参考资料

        # ② 生成：把参考资料塞进 system prompt，调 DeepSeek 生成答案
        resp = app.llm.invoke([
            SystemMessage(content=app.SYSTEM_PROMPT + "\n\n【参考资料】\n" + ctx_str),
            HumanMessage(content=q),
        ])

        # ③ 组装成 ragas 认识的样本记录
        records.append({
            "user_input": q,                 # 问题
            "retrieved_contexts": contexts,  # 检索上下文
            "response": resp.content,        # AI 答案
            "reference": s["ground_truth"],  # 参考答案
        })
        print(f"[{i}/{len(samples)}] {q[:20]}...")   # 进度提示

    return records


THRESHOLDS = {
    "faithfulness": 0.9,          # 忠实度：幻觉零容忍，门槛最高
    "answer_correctness": 0.8,    # 正确性：答错 = 误诊风险
    "context_recall": 0.8,        # 召回率：漏检关键信息 = 给不出正确依据
    "context_precision": 0.7,     # 精确率：冗余可容忍，但不能乱
    "answer_relevancy": 0.7,      # 相关性：跑题影响体验
}


def analyze_with_llm(df, llm) -> str:
    """
    企业级能力：把评测结果交给 LLM，生成一份可交付的分析报告。

    报告内容：总体评价 + 逐指标诊断 + 最差样本剖析 + 修复建议 + 下一步验证。
    这样测试/开发人员不用自己啃数字，直接看报告就知道问题在哪、怎么改。
    """
    score_cols = [c for c in THRESHOLDS if c in df.columns]
    if not score_cols:
        return "评测结果里没找到预期指标列，无法生成分析。请检查 ragas 版本。"

    # 各指标均值
    means = {c: round(float(df[c].mean()), 3) for c in score_cols}

    # 挑最差的 3 条样本做案例分析（优先按 faithfulness 排，因为它最致命）
    sort_key = "faithfulness" if "faithfulness" in df.columns else score_cols[0]
    worst = df.nsmallest(3, sort_key)

    cases = []
    for _, row in worst.iterrows():
        ctx = row.get("retrieved_contexts", [])
        if isinstance(ctx, list):
            ctx = "\n---\n".join(str(c) for c in ctx[:3])   # 最多取前 3 条检索内容
        cases.append({
            "问题": row.get("user_input", ""),
            "AI答案": row.get("response", ""),
            "参考答案": row.get("reference", ""),
            "检索到的内容": ctx,
            "各指标分数": {c: round(float(row.get(c, 0)), 3) for c in score_cols},
        })

    prompt = f"""你是资深 RAG 系统质量分析师，负责医疗问答系统的质量诊断。

以下是评测结果，请生成一份可交付给开发/测试人员的分析报告。

【各指标均值】
{json.dumps(means, ensure_ascii=False, indent=2)}

【医学场景及格线】
{json.dumps(THRESHOLDS, ensure_ascii=False, indent=2)}

【最差样本（用于案例分析）】
{json.dumps(cases, ensure_ascii=False, indent=2)}

请严格按以下结构输出 Markdown 报告：
## 一、总体评价
（2-3 句：系统整体处于什么水平，最突出的问题是什么）

## 二、逐指标分析
（每个指标单独一段：当前值 vs 及格线、是否达标、不达标可能的技术原因）

## 三、典型问题样本剖析
（针对最差样本，逐条判断问题类型：①检索没搜到关键内容 ②检索到了但生成答错 ③答案幻觉编造 ④答非所问。要引用样本里的实际内容作证据）

## 四、修复建议（按优先级排序）
（每条建议要具体可执行，例如：调整混合检索权重、增大检索 k、提高 RAG_MIN_SCORE 阈值、优化 grade 校验 prompt、改进数据清洗等）

## 五、下一步验证方式
（建议怎么验证修复是否有效）

要求：中文输出，直接给 Markdown，不要客套话和免责声明。"""

    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content


def main():
    # 命令行参数：--n 样本数、--seed 随机种子
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # ========== ragas 相关导入（放函数内，避免 import main 之前就加载 ragas） ==========
    from ragas import evaluate                          # ragas 的核心评测引擎
    from ragas.dataset_schema import EvaluationDataset  # 评测数据集容器
    from ragas.metrics import (                         # 五个评测指标
        Faithfulness, AnswerRelevancy, ContextPrecision,
        ContextRecall, AnswerCorrectness,
    )
    from ragas.llms import LangchainLLMWrapper              # 把 LangChain 的 LLM 包成 ragas 能用的
    from ragas.embeddings import LangchainEmbeddingsWrapper  # 把 embeddings 包成 ragas 能用的

    # 包装：ragas 不认识你的 DeepSeek / DashScope，必须用 wrapper 做适配
    evaluator_llm = LangchainLLMWrapper(app.llm)                     # 用 DeepSeek 当"打分裁判"
    evaluator_embeddings = LangchainEmbeddingsWrapper(app.embeddings)  # 用 DashScope embedding

    # 生成评测数据：抽样本 + 跑一遍检索生成（返回 list[dict] 给 from_dict）
    records=  run_pipeline(load_samples(args.n, args.seed))
    dataset = EvaluationDataset.from_dict(
      records
    )

    # 五个指标，各自衡量一个维度：
    #   Faithfulness      : 答案是否基于检索内容（防幻觉，医学场景最重要）
    #   AnswerRelevancy   : 答案是否相关
    #   ContextPrecision  : 检索精确率（检索到的内容里，相关的占比）
    #   ContextRecall     : 检索召回率（参考答案的关键信息，是否被检索到）
    #   AnswerCorrectness : 答案正确性（对照参考答案）

#     指标	区间	满分意味着	低分意味着（要警惕什么）
# Faithfulness	0~1	答案每个陈述都能被检索内容支持	答案在瞎编（幻觉），医学场景最危险
# AnswerRelevancy	0~1	答案完全紧扣问题	答非所问、跑题
# ContextPrecision	0~1	检索到的内容全部相关，且相关项排最前	检索回来了一堆无关内容（浪费 token 还误导）
# ContextRecall	0~1	参考答案的关键信息全被检索到	检索漏了关键信息（库里有但没搜出来）
# AnswerCorrectness	0~1	答案与参考答案事实+语义都一致	答案答错了


    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ]

    # 执行评测：ragas 对每条样本逐指标打分（内部会多次调 LLM 判断，所以慢）
    result = evaluate(dataset, metrics=metrics)

    # 结果转成表格，打印均值 + 存 CSV
    df = result.to_pandas()
    print("\n========== 各指标均值 ==========")
    print(df.mean(numeric_only=True).round(3))   # numeric_only 只对数字列求均值
    df.to_csv("ragas_results.csv", index=False, encoding="utf-8-sig")  # utf-8-sig 让 Excel 打开不乱码
    print("\n明细已保存 ragas_results.csv")

    # ========== 企业级：让 LLM 基于分数生成分析报告 ==========
    print("\n========== 正在生成分析报告（LLM 分析中，请稍候） ==========")
    report = analyze_with_llm(df, app.llm)
    print("\n" + report)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"ragas_report_{ts}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n分析报告已保存：{report_path}")


if __name__ == "__main__":   # 直接运行这个文件时才执行 main()；被 import 时不执行
    main()




# | 指标低 | 阈值参考 | 可能原因 | 优化方向 |
# | :--- | :--- | :--- | :--- |
# | **Context Recall** | < 0.75 | top_k 太小；chunk 太大；embedding 召回差 | 加大 top_k；缩小 chunk_size；换更强 embedding |
# | **Context Precision** | < 0.65 | 检索结果含大量无关片段 | 加 Reranker；提高相似度阈值；改善文档质量 |
# | **Answer Relevancy** | < 0.80 | Prompt 没有引导模型直接回答；答案太长绕弯 | Query 重写；优化 response prompt |
# | **Faithfulness** | < 0.75 | System prompt 约束弱；检索内容不足 | 强化 system prompt；先提高 Recall |

