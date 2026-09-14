"""检索命中率对照实验：量化「查询改写」对检索效果的真实提升。

注意（重要经验）：
    CrossEncoder 用 Sigmoid 激活，分数会**饱和**——残句也能打到 1.000，
    因此「rerank 分数」无法区分「查到对的内容」和「语义漂移」。
    本脚本改用更可靠的判据：**检索到的 top-k 文档是否真的包含前文实体**。

对同一批多轮追问用例，分别用两种查询检索：
    A) 关闭改写：直接用追问残句（复现语义漂移）
    B) 开启改写：用改写的完整查询

核心指标：
    实体命中率 = top-k 文档中包含「前文实体关键词」的比例

运行：
    & ".\.venv\Scripts\python.exe" retrieval_hit_rate_test.py
"""
from dotenv import load_dotenv

load_dotenv()

from doctor_agent.nodes.rewrite import rewrite_query_node
from doctor_agent.retrieval import get_retrieval, rerank_documents
from test_semantic_drift import DRIFT_CASES, build_messages


def probe(query: str, ensemble, keywords: list[str], top_k: int = 3):
    """返回 (top1 分数, 是否命中实体, 命中的关键词)。"""
    ranked = rerank_documents(query, ensemble.invoke(query))
    docs = [d for _, d in ranked][:top_k]
    top1 = ranked[0][0] if ranked else 0.0

    text = "\n".join(d.page_content for d in docs)
    hits = [k for k in keywords if k in text]
    return top1, bool(hits), hits


def main() -> None:
    ensemble, _ = get_retrieval()
    rows = []

    print("=" * 100)
    print("检索命中率对照实验：关闭改写 vs 开启改写")
    print("判据：检索到的 top-3 文档是否包含前文实体（而非 rerank 分数，分数会饱和）")
    print("=" * 100)

    for case in DRIFT_CASES:
        keywords = case["keywords"]
        msgs = build_messages(case)
        raw_q = case["followup"]
        new_q = rewrite_query_node({"messages": msgs})["query"]

        s_raw, hit_raw, hits_raw = probe(raw_q, ensemble, keywords)
        s_new, hit_new, hits_new = probe(new_q, ensemble, keywords)
        rows.append((case["name"], hit_raw, hit_new))

        print(f"\n[{case['name']}]  前文实体：{keywords}")
        print(
            f"  关闭改写：「{raw_q}」 → top1={s_raw:.3f} | "
            f"实体命中={'✅ ' + str(hits_raw) if hit_raw else '❌ 无'}"
        )
        print(
            f"  开启改写：「{new_q}」 → top1={s_new:.3f} | "
            f"实体命中={'✅ ' + str(hits_new) if hit_new else '❌ 无'}"
        )

    n = len(rows)
    hit_raw_cnt = sum(1 for r in rows if r[1])
    hit_new_cnt = sum(1 for r in rows if r[2])
    avg_raw = hit_raw_cnt / n
    avg_new = hit_new_cnt / n

    print("\n" + "=" * 100)
    print("汇总")
    print("=" * 100)
    print(f"用例数：{n}")
    print(f"实体命中率：关闭改写 = {hit_raw_cnt}/{n} = {avg_raw:.1%}")
    print(f"实体命中率：开启改写 = {hit_new_cnt}/{n} = {avg_new:.1%}")
    print(f"提升：{(avg_new - avg_raw) * 100:+.1f} 个百分点")


if __name__ == "__main__":
    main()
