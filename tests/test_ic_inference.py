"""
IC 显著性推断测试：Newey-West vs block bootstrap。

    python tests/test_ic_inference.py

覆盖：
    1. 重采样 —— 长度、可复现、环形首尾相接、平稳 bootstrap
    2. 白噪声 —— bootstrap SE 接近 std/sqrt(n)
    3. 正自相关 —— 块长越大 SE 越大，NW/boot 都比普通 t 收缩
    4. 置信区间 —— 强信号不跨 0，零均值信号大概率跨 0
    5. 边界 —— NaN、常量、单点、非法参数
    6. 对比对象 —— 默认块长、字段一致性
    7. 文本报告
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.analysis.ic_inference import (
    BootstrapStats,
    InferenceComparison,
    block_bootstrap_mean,
    circular_block_resample,
    compare_inference,
    format_inference_report,
    stationary_block_resample,
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
# 1. 重采样
# ===================================================================

def test_resample():
    print("\n── resampling ──")
    rng = np.random.default_rng(3)
    x = rng.normal(size=200)

    a = circular_block_resample(x, 5, n_boot=100, seed=7)
    b = circular_block_resample(x, 5, n_boot=100, seed=7)
    check(a.shape == (100,), "circular returns n_boot means")
    check(np.allclose(a, b), "same seed reproduces circular means")

    c = circular_block_resample(np.array([1.0, 2.0, 3.0, 4.0]),
                                4, n_boot=20, seed=1)
    check(np.allclose(c, 2.5), "one full-length block returns the sample mean")

    s1 = stationary_block_resample(x, 6, n_boot=100, seed=7)
    s2 = stationary_block_resample(x, 6, n_boot=100, seed=7)
    check(s1.shape == (100,), "stationary returns n_boot means")
    check(np.allclose(s1, s2), "same seed reproduces stationary means")

    check(circular_block_resample(x, 0, n_boot=10).size == 0,
          "block_size 0 returns empty")
    check(circular_block_resample([np.nan], 2, n_boot=10).size == 0,
          "single non-finite value returns empty")


# ===================================================================
# 2. 白噪声
# ===================================================================

def test_iid():
    print("\n── iid normal ──")
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, size=400)

    b = block_bootstrap_mean(x, block_size=1, n_boot=2000, seed=0)
    approx = float(x.std(ddof=1)) / np.sqrt(x.size)

    check(b.n_boot == 2000, "n_boot recorded")
    check(abs(b.se - approx) / approx < 0.15,
          f"bootstrap SE {b.se:.4f} near std/sqrt(n) {approx:.4f}")
    check(b.ci_low < b.mean < b.ci_high, "CI brackets the sample mean")
    check(0.0 <= b.p_two_sided <= 1.0, "p value in [0, 1]")
    check(abs(b.t_stat) < 4.0, "t statistic is not absurd on white noise")


# ===================================================================
# 3. 正自相关
# ===================================================================

def test_autocorrelation():
    print("\n── positive autocorrelation ──")
    x = ar1(600, 0.7, seed=5)

    se1 = block_bootstrap_mean(x, block_size=1, n_boot=1200, seed=0).se
    se20 = block_bootstrap_mean(x, block_size=20, n_boot=1200, seed=0).se
    check(se20 > se1 * 1.5,
          f"longer blocks inflate SE ({se1:.4f} -> {se20:.4f})")

    c = compare_inference(x, lags=5, block_size=5, n_boot=1200, seed=0)
    check(abs(c.t_nw) < abs(c.t_naive),
          f"Newey-West shrinks t ({c.t_naive:.2f} -> {c.t_nw:.2f})")
    check(abs(c.t_boot) < abs(c.t_naive),
          f"bootstrap shrinks t ({c.t_naive:.2f} -> {c.t_boot:.2f})")
    check(c.t_nw * c.t_boot > 0 or abs(c.t_nw) < 0.5,
          "NW and bootstrap agree on the direction")


# ===================================================================
# 4. 置信区间
# ===================================================================

def test_ci():
    print("\n── confidence interval ──")
    rng = np.random.default_rng(6)

    strong = rng.normal(1.0, 1.0, size=400)
    c = compare_inference(strong, lags=0, block_size=1,
                          n_boot=1500, seed=0)
    check(c.boot.p_two_sided < 0.01,
          f"strong mean is significant (p={c.boot.p_two_sided:.4f})")
    check(c.boot.ci_low > 0.0, "95% CI stays above zero")

    # 把样本均值强制归零，测的就是"样本均值为 0 时不会误报"
    null = rng.normal(0.0, 1.0, size=400)
    null -= null.mean()
    c0 = compare_inference(null, lags=0, block_size=1,
                           n_boot=1500, seed=0)
    check(c0.boot.ci_low < 0.0 < c0.boot.ci_high,
          "zero-mean sample CI crosses zero")
    check(c0.boot.p_two_sided > 0.05,
          f"zero-mean sample is not significant (p={c0.boot.p_two_sided:.4f})")


# ===================================================================
# 5. 边界
# ===================================================================

def test_edges():
    print("\n── edge cases ──")
    x = np.array([1.0, np.nan, 2.0, 3.0, np.inf])

    b = block_bootstrap_mean(x, block_size=2, n_boot=200, seed=0)
    b2 = block_bootstrap_mean([1.0, 2.0, 3.0], block_size=2,
                              n_boot=200, seed=0)
    check(np.isfinite(b.se) and b.n_boot == 200,
          "NaN/inf are dropped before resampling")

    short = block_bootstrap_mean([1.0], block_size=1, n_boot=100)
    check(short.n_boot == 0 and np.isnan(short.se),
          "single observation yields an empty bootstrap")

    const = block_bootstrap_mean([2.0] * 20, block_size=2, n_boot=100)
    check(const.se == 0.0 and const.t_stat == 0.0,
          "constant series has zero SE and t=0")

    c = compare_inference([], lags=1, n_boot=100)
    check(np.isnan(c.mean) and c.t_naive == 0.0 and c.boot.n_boot == 0,
          "empty series degrades gracefully")

    thrown = 0
    for fn in (lambda: block_bootstrap_mean(x, method="bogus"),
               lambda: block_bootstrap_mean(x, ci=1.5),
               lambda: compare_inference(x, lags=-1)):
        try:
            fn()
        except ValueError:
            thrown += 1
    check(thrown == 3, "invalid method / ci / negative lags raise ValueError")


# ===================================================================
# 6. 对比对象
# ===================================================================

def test_compare():
    print("\n── comparison object ──")
    x = ar1(300, 0.4, seed=9)

    c = compare_inference(x, lags=4, n_boot=500, seed=2)
    check(isinstance(c, InferenceComparison), "returns InferenceComparison")
    check(isinstance(c.boot, BootstrapStats), "embeds BootstrapStats")
    check(c.n == x.size, "sample size recorded")
    check(c.lags == 4, "lags recorded")
    check(c.boot.block_size == 5.0, "default block size is lags + 1")
    check(abs(c.t_boot - c.boot.t_stat) < 1e-12,
          "t_boot mirrors the bootstrap statistic")
    check(abs(c.boot_se - c.boot.se) < 1e-12,
          "boot_se mirrors the bootstrap SE")

    c2 = compare_inference(x, lags=4, block_size=12, n_boot=500, seed=2)
    check(c2.boot.block_size == 12.0, "explicit block size is kept")


# ===================================================================
# 7. 报告
# ===================================================================

def test_report():
    print("\n── report ──")
    x = ar1(200, 0.3, seed=1)
    results = {
        ("momentum_20", 1): compare_inference(x, lags=0, n_boot=200, seed=0),
        ("momentum_20", 5): compare_inference(x, lags=4, n_boot=200, seed=0),
    }
    text = format_inference_report(results, ["momentum_20"], [1, 5])

    for needle in ("Newey-West vs block bootstrap", "t_naive", "t_NW",
                   "t_boot", "95% CI", "momentum_20", "读法"):
        check(needle in text, f"report contains '{needle}'")


def main():
    print("=" * 62)
    print("  IC inference tests")
    print("=" * 62)

    test_resample()
    test_iid()
    test_autocorrelation()
    test_ci()
    test_edges()
    test_compare()
    test_report()

    print(f"\n  Result: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
