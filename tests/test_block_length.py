"""
Politis-White 自动块长测试。

    python tests/test_block_length.py

覆盖：
    1. 自相关 —— 定义、白噪声、AR(1) 衰减、边界
    2. 隐含假设检验 —— m_hat 自动选择与手动覆盖
    3. 块长公式 —— 白噪声夹到 1、持续性越强块长越长、n 越大块长越大、
                    SB 比 CB 短约 12.6%、不超 b_max
    4. ic_inference 接线 —— compare_inference 的 pw-sb / pw-cb
    5. 块长-SE 曲线
    6. 覆盖率模拟 —— 块长 1 明显低覆盖、自适应块长把覆盖率拉回来
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.analysis.block_length import (
    BlockLengthResult,
    politi_white_block_length,
    sample_acf,
)
from src.analysis.ic_inference import (
    bootstrap_se_curve,
    compare_inference,
    coverage_simulation,
    format_coverage_report,
)

PASS = 0
FAIL = 0


def check(cond, name):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def ar1(n, phi, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# ===================================================================
# 1. 自相关
# ===================================================================

def test_acf():
    print("\n── sample autocorrelation ──")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    acf = sample_acf(x, 2)
    check(acf.size == 3, "returns max_lag + 1 values")
    check(abs(acf[0] - 1.0) < 1e-12, "lag 0 is exactly 1")

    # 线性序列 [1..5] 的样本自相关：rho(1) = 4/10 = 0.4
    check(abs(acf[1] - 0.4) < 1e-12, "linear series lag-1 acf = 0.4")

    np.random.seed(0)
    noise = np.random.randn(4000)
    nz = sample_acf(noise, 3)
    check(all(abs(v) < 0.05 for v in nz[1:]),
          "white noise acf is near zero")

    ar = ar1(3000, 0.6, seed=2)
    aa = sample_acf(ar, 5)
    check(aa[1] > 0.5 and aa[5] < aa[1],
          "AR(1) acf decays with lag")

    check(sample_acf([1.0], 5).size == 1, "single point clamps max_lag")
    check(np.all(sample_acf([2.0] * 10, 3) == 0.0) or
          sample_acf([2.0] * 10, 3)[0] == 1.0,
          "constant series is handled")


# ===================================================================
# 2. 隐含假设检验
# ===================================================================

def test_m_hat():
    print("\n── implied hypothesis test ──")
    np.random.seed(3)
    noise = np.random.randn(500)
    r = politi_white_block_length(noise)
    check(r.m_hat == 0 and r.max_lag == 0,
          "white noise: m_hat = 0")

    ar = ar1(2000, 0.7, seed=4)
    r_auto = politi_white_block_length(ar)
    check(r_auto.m_hat > 0, "AR(1): m_hat > 0")
    check(r_auto.max_lag == 2 * r_auto.m_hat, "M = 2 * m_hat")

    r_manual = politi_white_block_length(ar, m_hat=1)
    check(r_manual.m_hat == 1 and r_manual.max_lag == 2,
          "manual m_hat overrides the test")

    check(politi_white_block_length(ar, m_hat=10 ** 6).m_hat <=
          int(np.ceil(np.sqrt(2000))) + 5, "m_hat is capped at m_max")


# ===================================================================
# 3. 块长公式
# ===================================================================

def test_block_length():
    print("\n── block length formula ──")
    np.random.seed(5)
    noise = np.random.randn(1000)
    check(politi_white_block_length(noise).b_sb == 1.0,
          "white noise clips to 1")

    lengths = []
    for rho in (0.2, 0.5, 0.8):
        r = politi_white_block_length(ar1(2000, rho, seed=6))
        lengths.append(r.b_sb)
    check(lengths[0] < lengths[1] < lengths[2],
          f"persistence increases block length {lengths}")

    b_small = politi_white_block_length(ar1(300, 0.5, seed=7)).b_sb
    b_large = politi_white_block_length(ar1(3000, 0.5, seed=7)).b_sb
    check(b_large > b_small, "longer series picks longer block")

    r = politi_white_block_length(ar1(1500, 0.5, seed=8))
    check(1.0 <= r.b_sb <= r.b_max and 1.0 <= r.b_cb <= r.b_max,
          "both lengths are clipped to [1, b_max]")
    check(r.b_cb > r.b_sb, "circular block is longer than stationary")
    ratio = r.b_sb / r.b_cb
    check(abs(ratio - (2.0 / 3.0) ** (1.0 / 3.0)) < 0.01,
          "SB/CB ratio is (2/3)^(1/3)")

    tiny = politi_white_block_length([1.0, 2.0, 3.0])
    check(tiny.b_sb == 1.0 and tiny.n == 3, "short series degrades to 1")
    const = politi_white_block_length([3.0] * 100)
    check(const.b_sb == 1.0 and const.g_hat == 0.0,
          "constant series gives block 1")
    check(isinstance(const, BlockLengthResult), "returns dataclass")


# ===================================================================
# 4. compare_inference 接线
# ===================================================================

def test_compare_wiring():
    print("\n── compare_inference wiring ──")
    x = ar1(800, 0.5, seed=9)
    pw = politi_white_block_length(x)

    c = compare_inference(x, lags=4, block_method="pw-sb",
                          n_boot=400, seed=0)
    check(c.block_source == "pw-sb", "source recorded as pw-sb")
    check(abs(c.block_length - pw.b_sb) < 1e-12,
          "pw-sb uses the stationary block length")

    c2 = compare_inference(x, lags=4, block_method="pw-cb",
                           n_boot=400, seed=0)
    check(abs(c2.block_length - pw.b_cb) < 1e-12,
          "pw-cb uses the circular block length")

    c3 = compare_inference(x, lags=4, block_method="horizon",
                           n_boot=400, seed=0)
    check(c3.block_length == 5.0, "horizon keeps lags + 1")

    c4 = compare_inference(x, lags=4, block_size=3, block_method="pw-sb",
                           n_boot=400, seed=0)
    check(c4.block_length == 3.0, "explicit block_size wins")

    try:
        compare_inference(x, lags=4, block_method="nope", n_boot=100)
        check(False, "invalid block_method raises")
    except ValueError:
        check(True, "invalid block_method raises")


# ===================================================================
# 5. 块长-SE 曲线
# ===================================================================

def test_se_curve():
    print("\n── block length / SE curve ──")
    x = ar1(1200, 0.6, seed=10)
    sizes = [1, 5, 20, 60]
    rows = bootstrap_se_curve(x, sizes, n_boot=400, seed=1)
    check(len(rows) == len(sizes), "one row per block size")
    check([r["block_size"] for r in rows] == [1.0, 5.0, 20.0, 60.0],
          "block sizes preserved")
    check(rows[0]["se"] < rows[1]["se"],
          "SE grows once the block covers the autocorrelation")
    check(all(r["ci_low"] <= r["ci_high"] for r in rows),
          "CIs are ordered")
    check(bootstrap_se_curve(x, [], n_boot=100) == [],
          "empty grid returns no rows")


# ===================================================================
# 6. 覆盖率模拟
# ===================================================================

def test_coverage():
    print("\n── coverage simulation ──")
    sim = coverage_simulation(rho=0.6, n=600, reps=60, n_boot=300, seed=2)
    d = {row["label"]: row["coverage"] for row in sim.rows}

    check(abs(d["普通 t"] - d["L=1"]) < 0.10,
          "naive t and L=1 coverage are close")
    check(d["L=1"] < 0.85, f"L=1 undercovers ({d['L=1']:.0%})")
    check(d["PW 自适应"] > d["L=1"],
          "PW improves over single-point bootstrap")
    check(d["PW 自适应"] >= 0.85,
          f"PW coverage is near nominal ({d['PW 自适应']:.0%})")
    check(sim.n == 600 and sim.reps == 60, "settings recorded")
    check(1.5 < sim.rows[4]["mean_block"] < sim.n / 3,
          "PW average block is inside the feasible range")

    text = format_coverage_report(sim)
    for needle in ("覆盖率", "PW 自适应", "L=1", "普通 t", "Newey-West"):
        check(needle in text, f"report contains '{needle}'")

    try:
        coverage_simulation(rho=1.0, n=100, reps=20, n_boot=100)
        check(False, "rho = 1 raises")
    except ValueError:
        check(True, "rho = 1 raises")
    try:
        coverage_simulation(rho=0.3, n=10, reps=20, n_boot=100)
        check(False, "tiny n raises")
    except ValueError:
        check(True, "tiny n raises")


def main():
    print("=" * 62)
    print("  Politis-White block length tests")
    print("=" * 62)

    test_acf()
    test_m_hat()
    test_block_length()
    test_compare_wiring()
    test_se_curve()
    test_coverage()

    print(f"\n  Result: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
