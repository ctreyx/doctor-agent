"""Enterprise cache validation suite for doctor_agent.retrieval_cache.

目标：
1) 覆盖检索缓存的核心功能（>=10 用例）
2) 在不依赖外部服务的前提下，稳定复现实验
3) 增加风险探针，识别当前实现的潜在漏洞

运行：
    & ".\\.venv\\Scripts\\python.exe" cache_test_suite.py
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from doctor_agent import config
from doctor_agent import retrieval_cache as rc


class FakeVec:
    """最小向量对象，支持 `@` 点积运算。"""

    def __init__(self, values: list[float]):
        self.values = values

    def __matmul__(self, other: "FakeVec") -> float:
        return float(sum(a * b for a, b in zip(self.values, other.values)))


class FakeEmbedder:
    """可控 embedding，保证测试可重复。"""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, text: str, normalize_embeddings: bool = True) -> FakeVec:  # noqa: ARG002
        self.calls += 1
        key = rc.normalize_query(text)
        mapping = {
            "感冒吃什么药": FakeVec([1.0, 0.0, 0.0]),
            "感冒如何治疗": FakeVec([0.98, 0.02, 0.0]),
            "肺癌早期症状": FakeVec([0.0, 1.0, 0.0]),
            "今天天气怎么样": FakeVec([0.0, 0.0, 1.0]),
        }
        return mapping.get(key, FakeVec([0.4, 0.4, 0.2]))


@dataclass
class TestCase:
    name: str
    fn: Callable[[], str]


@dataclass
class TestResult:
    name: str
    status: str
    detail: str


def _md_escape(text: str) -> str:
    """Escape markdown table special chars for stable rendering."""
    return text.replace("|", "\\|").replace("\n", "<br>")


def write_markdown_report(
    control_results: list[TestResult],
    mandatory_results: list[TestResult],
    probe_results: list[TestResult],
    control_passed: int,
    control_failed: int,
    passed: int,
    failed: int,
) -> Path:
    """Write enterprise-style markdown report and return output path."""
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"cache_test_report_{ts}.md"

    lines: list[str] = []
    lines.append("# RAG 检索缓存企业级测试报告")
    lines.append("")
    lines.append(f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 控制用例通过: {control_passed}")
    lines.append(f"- 控制用例失败: {control_failed}")
    lines.append(f"- 强制用例总数: {len(mandatory_results)}")
    lines.append(f"- 强制用例通过: {passed}")
    lines.append(f"- 强制用例失败: {failed}")
    lines.append(f"- 最终结果: {'通过' if (failed == 0 and control_failed == 0) else '失败'}")
    lines.append("")

    lines.append("## 缓存开关控制")
    lines.append("")
    lines.append("| 控制项 | 状态 | 结果 |")
    lines.append("|---|---|---|")
    for r in control_results:
        lines.append(
            f"| {_md_escape(r.name)} | {_md_escape(r.status)} | {_md_escape(r.detail)} |"
        )
    lines.append("")

    lines.append("## 强制用例")
    lines.append("")
    lines.append("| 用例 | 状态 | 缓存结果 |")
    lines.append("|---|---|---|")
    for r in mandatory_results:
        lines.append(
            f"| {_md_escape(r.name)} | {_md_escape(r.status)} | {_md_escape(r.detail)} |"
        )
    lines.append("")

    lines.append("## 风险探针")
    lines.append("")
    lines.append("| 探针 | 状态 | 细节 |")
    lines.append("|---|---|---|")
    for r in probe_results:
        lines.append(
            f"| {_md_escape(r.name)} | {_md_escape(r.status)} | {_md_escape(r.detail)} |"
        )
    lines.append("")

    lines.append("## 结论")
    lines.append("")
    if failed == 0:
        lines.append("所有强制用例通过。当前缓存行为在功能层面有效。")
    else:
        lines.append("存在强制用例失败，建议修复后再进入生产环境。")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


@contextmanager
def patch_config(**kwargs):
    backup = {k: getattr(config, k) for k in kwargs}
    try:
        for k, v in kwargs.items():
            setattr(config, k, v)
        yield
    finally:
        for k, v in backup.items():
            setattr(config, k, v)


def reset_cache_state() -> None:
    rc.invalidate_cache()
    rc._embedder = None


def control_case_cache_on() -> TestResult:
    """控制用例：启用缓存，验证写入后可命中。"""
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(RAG_CACHE_ENABLED=True, RAG_CACHE_TTL=3600, RAG_CACHE_MAX_SIZE=16):
        rc.set_cached("感冒如何治疗", "ctx-on", 0.93)
        got = rc.get_cached("感冒如何治疗")
    ok = got == ("ctx-on", 0.93, "exact")
    return TestResult(
        "缓存开启",
        "通过" if ok else "失败",
        f"缓存结果={got}",
    )


def control_case_cache_off() -> TestResult:
    """控制用例：关闭缓存，验证写入后不可命中。"""
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(RAG_CACHE_ENABLED=False, RAG_CACHE_TTL=3600, RAG_CACHE_MAX_SIZE=16):
        rc.set_cached("感冒如何治疗", "ctx-off", 0.93)
        got = rc.get_cached("感冒如何治疗")
    ok = got is None
    return TestResult(
        "缓存关闭",
        "通过" if ok else "失败",
        f"缓存结果={got}",
    )


def run_control_cases(mode: str) -> list[TestResult]:
    """按模式运行缓存开关控制用例。"""
    results: list[TestResult] = []
    if mode in ("on", "both"):
        results.append(control_case_cache_on())
    if mode in ("off", "both"):
        results.append(control_case_cache_off())
    return results


# ---------------------------
# Mandatory tests (must pass)
# ---------------------------
def test_01_normalize_query() -> str:
    got = rc.normalize_query("  感冒   如何   治疗  ")
    assert got == "感冒 如何 治疗"
    return f"归一化结果={got}"


def test_02_disabled_cache_bypass() -> str:
    reset_cache_state()
    with patch_config(RAG_CACHE_ENABLED=False):
        rc.set_cached("a", "ctx", 0.99)
        got = rc.get_cached("a")
        size = rc.cache_stats()["size"]
        assert got is None
        assert size == 0
        return f"缓存结果={got}, 容量={size}"


def test_03_exact_hit() -> str:
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(RAG_CACHE_ENABLED=True, RAG_CACHE_TTL=3600, RAG_CACHE_MAX_SIZE=16):
        rc.set_cached("肺癌早期症状", "ctx-1", 0.91)
        got = rc.get_cached("肺癌早期症状")
        assert got == ("ctx-1", 0.91, "exact")
        return f"缓存命中结果={got}"


def test_04_empty_query_noop() -> str:
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(RAG_CACHE_ENABLED=True):
        got = rc.get_cached("")
        rc.set_cached("", "ctx", 0.9)
        size = rc.cache_stats()["size"]
        assert got is None
        assert size == 0
        return f"缓存结果={got}, 容量={size}"


def test_05_ttl_expire() -> str:
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(RAG_CACHE_ENABLED=True, RAG_CACHE_TTL=-1, RAG_CACHE_MAX_SIZE=16):
        rc.set_cached("肺癌早期症状", "ctx-expire", 0.88)
        got = rc.get_cached("肺癌早期症状")
        assert got is None
        return f"缓存结果={got}"


def test_06_lru_evict_oldest() -> str:
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(RAG_CACHE_ENABLED=True, RAG_CACHE_TTL=3600, RAG_CACHE_MAX_SIZE=2):
        rc.set_cached("k1", "ctx1", 0.8)
        rc.set_cached("k2", "ctx2", 0.8)
        _ = rc.get_cached("k1")  # k1 变为最近使用，k2 成最老
        rc.set_cached("k3", "ctx3", 0.8)
        got_k2 = rc.get_cached("k2")
        got_k1 = rc.get_cached("k1")
        assert got_k2 is None
        assert got_k1 == ("ctx1", 0.8, "exact")
        return f"被淘汰键k2={got_k2}, 保留键k1={got_k1}"


def test_07_update_existing_key_no_growth() -> str:
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(RAG_CACHE_ENABLED=True, RAG_CACHE_TTL=3600, RAG_CACHE_MAX_SIZE=4):
        rc.set_cached("dup", "ctx-old", 0.5)
        rc.set_cached("dup", "ctx-new", 0.9)
        stats = rc.cache_stats()
        got = rc.get_cached("dup")
        assert stats["size"] == 1
        assert got == ("ctx-new", 0.9, "exact")
        return f"缓存命中结果={got}, 容量={stats['size']}"


def test_08_semantic_hit_paraphrase() -> str:
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(
        RAG_CACHE_ENABLED=True,
        RAG_CACHE_TTL=3600,
        RAG_CACHE_MAX_SIZE=16,
        RAG_CACHE_SEMANTIC_THRESHOLD=0.9,
    ):
        rc.set_cached("感冒吃什么药", "ctx-cold", 0.95)
        got = rc.get_cached("感冒如何治疗")
        assert got == ("ctx-cold", 0.95, "semantic")
        return f"语义命中结果={got}"


def test_09_semantic_miss_below_threshold() -> str:
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(
        RAG_CACHE_ENABLED=True,
        RAG_CACHE_TTL=3600,
        RAG_CACHE_MAX_SIZE=16,
        RAG_CACHE_SEMANTIC_THRESHOLD=0.99,
    ):
        rc.set_cached("感冒吃什么药", "ctx-cold", 0.95)
        got = rc.get_cached("感冒如何治疗")
        assert got is None
        return f"语义未命中结果={got}"


def test_10_invalidate_reset_stats() -> str:
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(RAG_CACHE_ENABLED=True, RAG_CACHE_TTL=3600, RAG_CACHE_MAX_SIZE=16):
        rc.set_cached("x", "ctx-x", 0.7)
        _ = rc.get_cached("x")
        assert rc.cache_stats()["hits"] >= 1
        rc.invalidate_cache()
        stats = rc.cache_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        return f"失效后统计={stats}"


def test_11_whitespace_equivalent_exact_hit() -> str:
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(RAG_CACHE_ENABLED=True, RAG_CACHE_TTL=3600, RAG_CACHE_MAX_SIZE=16):
        rc.set_cached("  肺癌   早期  症状  ", "ctx-space", 0.9)
        got = rc.get_cached("肺癌  早期 症状")
        assert got == ("ctx-space", 0.9, "exact")
        return f"空白归一命中结果={got}"


def test_12_chinese_exact_hit_again() -> str:
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(RAG_CACHE_ENABLED=True, RAG_CACHE_TTL=3600, RAG_CACHE_MAX_SIZE=16):
        rc.set_cached("普通感冒治疗", "ctx-普通感冒", 0.86)
        got = rc.get_cached("普通感冒治疗")
        assert got == ("ctx-普通感冒", 0.86, "exact")
        return f"中文精确命中结果={got}"


# ---------------------------
# Risk probes (non-blocking)
# ---------------------------
def probe_01_empty_cache_triggers_embedding() -> tuple[bool, str]:
    """理想行为：空缓存时不应调用 embedding；当前实现会调用。"""
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(RAG_CACHE_ENABLED=True):
        _ = rc.get_cached("任意问题")
    ok = fake.calls == 0
    msg = f"期望 embedding 调用次数=0，实际={fake.calls}"
    return ok, msg


def probe_02_low_score_can_be_cached() -> tuple[bool, str]:
    """理想行为：低分结果不应缓存；当前 set_cached 无分数门槛。"""
    reset_cache_state()
    fake = FakeEmbedder()
    rc._get_embedder = lambda: fake
    with patch_config(RAG_CACHE_ENABLED=True, RAG_CACHE_TTL=3600, RAG_CACHE_MAX_SIZE=16):
        rc.set_cached("低分测试", "ctx-low", 0.01)
        got = rc.get_cached("低分测试")
    ok = got is None
    msg = f"期望低分不缓存，实际命中={got}"
    return ok, msg


def run() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索缓存企业级测试")
    parser.add_argument(
        "--cache-mode",
        choices=["on", "off", "both"],
        default="both",
        help="缓存开关控制用例：on=仅缓存开启，off=仅缓存关闭，both=两者都跑（默认）",
    )
    args = parser.parse_args()

    tests = [
        TestCase("01 查询归一化", test_01_normalize_query),
        TestCase("02 关闭开关绕过缓存", test_02_disabled_cache_bypass),
        TestCase("03 精确键命中", test_03_exact_hit),
        TestCase("04 空查询不写不读", test_04_empty_query_noop),
        TestCase("05 TTL 过期失效", test_05_ttl_expire),
        TestCase("06 LRU 淘汰最旧项", test_06_lru_evict_oldest),
        TestCase("07 同键覆盖不扩容", test_07_update_existing_key_no_growth),
        TestCase("08 语义同义命中", test_08_semantic_hit_paraphrase),
        TestCase("09 阈值严格时语义未命中", test_09_semantic_miss_below_threshold),
        TestCase("10 失效后统计重置", test_10_invalidate_reset_stats),
        TestCase("11 空白归一仍命中", test_11_whitespace_equivalent_exact_hit),
        TestCase("12 中文精确命中（补充）", test_12_chinese_exact_hit_again),
    ]

    print("=" * 80)
    print("RAG 检索缓存企业级测试")
    print("=" * 80)

    print("缓存开关控制:")
    control_results = run_control_cases(args.cache_mode)
    for r in control_results:
        print(f"[{r.status}] {r.name} | {r.detail}")
    if not control_results:
        print("[提示] 当前模式未运行任何控制用例")
    control_passed = sum(1 for r in control_results if r.status == "通过")
    control_failed = sum(1 for r in control_results if r.status != "通过")
    print(f"控制用例统计: 通过={control_passed}, 失败={control_failed}, 总数={len(control_results)}")
    print("说明: 强制用例用于验证缓存功能逻辑，cache-mode 仅控制“开/关”模式对照验证。")
    print("-" * 80)

    passed = 0
    failed = 0
    mandatory_results: list[TestResult] = []
    for t in tests:
        try:
            result = t.fn()
            print(f"[通过] {t.name} | {result}")
            passed += 1
            mandatory_results.append(TestResult(t.name, "通过", result))
        except Exception as e:  # noqa: BLE001
            print(f"[失败] {t.name}: {e}")
            failed += 1
            mandatory_results.append(TestResult(t.name, "失败", str(e)))

    print("-" * 80)
    print(f"强制用例统计: 通过={passed}, 失败={failed}, 总数={len(tests)}")

    print("-" * 80)
    print("风险探针（不阻断）:")
    p1_ok, p1_msg = probe_01_empty_cache_triggers_embedding()
    p2_ok, p2_msg = probe_02_low_score_can_be_cached()
    print(f"[{'通过' if p1_ok else '风险'}] 探针01 空缓存仍触发embedding -> {p1_msg}")
    print(f"[{'通过' if p2_ok else '风险'}] 探针02 低分结果仍可缓存 -> {p2_msg}")
    probe_results = [
        TestResult(
            "探针01 空缓存仍触发embedding",
            "通过" if p1_ok else "风险",
            p1_msg,
        ),
        TestResult(
            "探针02 低分结果仍可缓存",
            "通过" if p2_ok else "风险",
            p2_msg,
        ),
    ]

    report_path = write_markdown_report(
        control_results=control_results,
        mandatory_results=mandatory_results,
        probe_results=probe_results,
        control_passed=control_passed,
        control_failed=control_failed,
        passed=passed,
        failed=failed,
    )
    print(f"Markdown 报告: {report_path}")

    print("=" * 80)
    if failed == 0 and control_failed == 0:
        print("结果: 通过（所有强制用例均通过）")
        return 0

    print("结果: 失败（存在控制用例或强制用例失败）")
    return 1


if __name__ == "__main__":
    raise SystemExit(run())
