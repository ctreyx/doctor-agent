"""语义漂移测试：证明/验证多轮追问的指代消解是否生效。

测试什么：
  多轮对话里，追问句往往是带指代的残句（"那会持续多久？"），
  直接拿去检索会"漂移"（搜不到对的内容）。
  系统应该先用 rewrite 节点把它改写成完整查询（"化疗后恶心会持续多久？"）。

两个测试：
  1. test_rewrite_node   —— 单元测试：直接测改写逻辑（快）
  2. test_graph_query    —— 集成测试：跑完整图，证明 rewrite 真正接入并生效（慢）

用法（在项目根目录执行）：
    .venv/Scripts/python.exe test_semantic_drift.py
"""
from dotenv import load_dotenv

load_dotenv()

import main as app
from langchain_core.messages import HumanMessage, AIMessage




DRIFT_CASES = [
    # ---- 1. 指示代词「那」指代 ----
    {
        "name": "指示代词「那」",
        "history": [
            ("user", "化疗后恶心怎么办？"),
            ("ai", "化疗后恶心常见，可少食多餐缓解。"),
        ],
        "followup": "那会持续多久？",
        "keywords": ["恶心"],          # 改写后必须包含「恶心」等前文实体
    },

    # ---- 2. 省略主语（追问只说了"有什么副作用"，没提什么药）----
    # {
    #     "name": "省略主语",
    #     "history": [
    #         ("user", "高血压吃什么药比较好？"),
    #         ("ai", "常用降压药有氨氯地平等。"),
    #     ],
    #     "followup": "有什么副作用？",
    #     "keywords": ["降压药"],
    # },

    # ---- 3. 「这个药」指代 ----
    {
        "name": "「这个药」指代",
        "history": [
            ("user", "阿司匹林应该怎么吃？"),
            ("ai", "阿司匹林通常每日一次，饭后服用。"),
        ],
        "followup": "这个药能长期吃吗？",
        "keywords": ["阿司匹林"],
    },

    # # ---- 4. 「它」指代 ----
    {
        "name": "「它」指代",
        "history": [
            ("user", "糖尿病饮食要注意什么？"),
            ("ai", "糖尿病患者要控制糖分摄入。"),
        ],
        "followup": "它能根治吗？",
        "keywords": ["糖尿病"],
    },

    # # ---- 5. 数量追问（省略了药名）----
    # {
    #     "name": "数量追问省略对象",
    #     "history": [
    #         ("user", "布洛芬退烧一次吃多少？"),
    #         ("ai", "成人一次 200-400 毫克。"),
    #     ],
    #     "followup": "一天最多几次？",
    #     "keywords": ["布洛芬"],
    # },

    # # ---- 6. 并列追问（省略前文主体）----
    # {
    #     "name": "并列追问省略主体",
    #     "history": [
    #         ("user", "化疗后白细胞低怎么办？"),
    #         ("ai", "可通过升白针或饮食调理。"),
    #     ],
    #     "followup": "还有哪些注意事项？",
    #     "keywords": ["白细胞"],
    # },

    # # ---- 7. 时间追问 ----
    # {
    #     "name": "时间追问",
    #     "history": [
    #         ("user", "手术后多久能拆线？"),
    #         ("ai", "一般 7-14 天拆线。"),
    #     ],
    #     "followup": "那多久能恢复正常活动？",
    #     "keywords": ["手术"],
    # },

    # # ---- 8. 饮食追问（省略疾病）----
    # {
    #     "name": "饮食追问省略疾病",
    #     "history": [
    #         ("user", "胃炎患者饮食要注意什么？"),
    #         ("ai", "避免辛辣刺激食物。"),
    #     ],
    #     "followup": "能吃辣的吗？",
    #     "keywords": ["胃炎"],
    # },

    # # ---- 9. 复发追问 ----
    # {
    #     "name": "复发追问",
    #     "history": [
    #         ("user", "乳腺癌术后会复发吗？"),
    #         ("ai", "存在复发可能，需定期随访。"),
    #     ],
    #     "followup": "复发率有多高？",
    #     "keywords": ["乳腺癌"],
    # },

    # # ---- 10. 用药相互作用（追问省略药名）----
    # {
    #     "name": "用药相互作用省略药名",
    #     "history": [
    #         ("user", "我在吃华法林，有什么要注意的？"),
    #         ("ai", "需定期监测凝血功能。"),
    #     ],
    #     "followup": "能和其他药一起吃吗？",
    #     "keywords": ["华法林"],
    # },

    # # ---- 11. 复查频率追问 ----
    # {
    #     "name": "复查频率追问",
    #     "history": [
    #         ("user", "甲状腺结节需要做哪些检查？"),
    #         ("ai", "通常做甲状腺超声和甲功。"),
    #     ],
    #     "followup": "多久复查一次？",
    #     "keywords": ["甲状腺结节"],
    # },

    # # ---- 12. 症状追问（省略疾病）----
    # {
    #     "name": "症状追问省略疾病",
    #     "history": [
    #         ("user", "感冒咳嗽厉害怎么办？"),
    #         ("ai", "多喝水，必要时用止咳药。"),
    #     ],
    #     "followup": "还有什么别的症状？",
    #     "keywords": ["感冒"],
    # },
]

def build_messages(case: dict) -> list:
    """把用例的 history 转成消息列表，并 append 追问句（追问必须是最后一条）。"""
    msgs = []
    for role, text in case["history"]:
        msgs.append(HumanMessage(text) if role == "user" else AIMessage(text))
    msgs.append(HumanMessage(case["followup"]))
    return msgs


# ========== 单元测试：直接测改写逻辑（不跑完整图，快） ==========

def test_rewrite_node():
    """逐条用例测改写：追问应被消解为包含前文实体的完整查询。"""
    passed, failed = 0, 0

    for case in DRIFT_CASES:
        state = {"messages": build_messages(case)}
        query = app.rewrite_query_node(state)["query"]
        print(f"[{case['name']}] 追问：{case['followup']!r}")
        print(f"             改写：{query}")

        # 断言：keywords 里【至少一个】前文实体出现在改写结果中
        hit = [k for k in case["keywords"] if k in query]
        if hit:
            passed += 1
            print(f"             ✅ 命中 {hit}")
        else:
            failed += 1
            print(f"             ❌ 未命中任何关键词 {case['keywords']}")

    print(f"\n指代消解成功率：{passed}/{passed + failed}")
    assert failed == 0, f"❌ 有 {failed} 条用例语义漂移未解决"


# ========== 集成测试：跑完整图，证明 rewrite 真正接入 ==========
# def test_graph_query():
#     """完整图跑多轮：state['query'] 应是完整查询（证明 rewrite 接入且生效）。"""
#     messages = [
#         HumanMessage("化疗后恶心怎么办？"),
#         AIMessage("化疗后恶心常见，可少食多餐缓解。"),
#         HumanMessage("那会持续多久？"),
#     ]
#     result = app.graph.invoke(
#         {"messages": messages},
#         config={"configurable": {"thread_id": "drift-integration"}},
#     )
#     query = result.get("query", "")
#     print(f"state['query']：{query}")

#     # 断言 1：state 里必须有 query —— 若没有，说明 rewrite 没被执行/没接入图
#     assert query, "❌ state 里没有 query（rewrite 节点没接入图，直接 guard→retrieve 了）"
#     # 断言 2：query 必须是完整句，不能是残句
#     assert "化疗" in query or "恶心" in query, f"❌ 语义漂移未解决，查询是残句：'{query}'"
#     print("✅ 集成测试通过：rewrite 已接入图并生效\n")


def test_graph_query():
    passed, failed = 0, 0
    for i, case in enumerate(DRIFT_CASES):
        result = app.graph.invoke(
            {"messages": build_messages(case)},
            config={"configurable": {"thread_id": f"drift-graph-{i}"}},
        )
        query = result.get("query", "")

        hit = [k for k in case["keywords"] if k in query]

        # 每条都打印：期望 vs 实际
        print(f"[{case['name']}]")
        print(f"    追问：{case['followup']!r}")
        print(f"    期望关键词：{case['keywords']}")
        print(f"    实际改写：{query}")
        print(f"    命中：{hit if hit else '无'}   {'✅' if hit else '❌'}")

        if hit:
            passed += 1
        else:
            failed += 1

    print(f"\n图内指代消解成功率：{passed}/{passed + failed}")
    assert failed == 0, f"❌ 有 {failed} 条在图内漂移未解决"


def run_graph_cases(tag: str) -> int:
    """跑图，返回「未命中关键词」的条数。"""
    failed = 0
    for i, case in enumerate(DRIFT_CASES):
        result = app.graph.invoke(
            {"messages": build_messages(case)},
            config={"configurable": {"thread_id": f"{tag}-{i}"}},
        )
        query = result.get("query", "")
        answer = result.get("generation", "")          # ← 最终答案
        hit = [k for k in case["keywords"] if k in query]

        print(f"[{case['name']}]")
        print(f"    实际query：{query!r}  命中：{hit if hit else '无'}  {'✅' if hit else '❌'}")
        print(f"    实际回答：{answer}")               # ← 打印答案
        print()

        if not hit:
            failed += 1
    return failed


def test_with_rewrite():
    """开 rewrite：语义漂移被解决 → 应全绿（通过）。"""
    app.ENABLE_REWRITE = True
    failed = run_graph_cases("rewrite-on")
    print(f"\n【开 rewrite】图内消解成功率：{len(DRIFT_CASES) - failed}/{len(DRIFT_CASES)}")
    assert failed == 0, f"❌ 开 rewrite 仍有 {failed} 条漂移未解决"


def test_without_rewrite():
    """关 rewrite：语义漂移复现 → 应全红（失败，证明漂移存在）。"""
    app.ENABLE_REWRITE = False
    failed = run_graph_cases("rewrite-off")
    print(f"\n【关 rewrite】图内消解成功率：{len(DRIFT_CASES) - failed}/{len(DRIFT_CASES)}")
    assert failed == 0, f"❌ 关 rewrite 后 {failed} 条漂移未解决"   # ← 这行会失败，这正是你要的「红」


if __name__ == "__main__":
    # test_rewrite_node()
    # test_graph_query()   # 集成测试太慢，可注释掉，先跑快的
    # print("🎉 全部通过，语义漂移已解决")



    print("===== 对照实验 1：开 rewrite（预期全绿）=====")
    test_with_rewrite()
    # print("\n===== 对照实验 2：关 rewrite（预期全红）=====")
    # test_without_rewrite()
