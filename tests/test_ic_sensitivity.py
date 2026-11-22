"""
IC 期限敏感性测试。

    python tests/test_ic_sensitivity.py

覆盖：
    1. Newey-West t —— 退化、白噪声、正自相关、非法参数
    2. 因子构造 —— momentum/reversal 方向、非法期限
    3. 单格统计 —— 与 compute_ic 一致、空样本
    4. 两个方向的扫描 —— 形状、去重、未知因子
    5. 交叉网格 —— 与单独扫描逐格一致
    6. 宽表整理 —— 嵌套/扁平两种输入、缺格 NaN
    7. 分段稳定性 —— 分段数、同号占比
    8. 无未来信息 —— 截断尾部不影响重叠区间的 IC
    9. 文本报告 —— 关键小节齐全
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.analysis.factor_research import compute_forward_returns, compute_ic
from src.analysis.ic_sensitivity import (
    ICStats,
    analyze_grid,
    analyze_horizons,
    analyze_lookbacks,
    format_sensitivity_report,
    ic_stats,
    make_lookback_factor,
    newey_west_tstat,
    sign_consistency,
    stats_table,
    subperiod_ic,
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


def make_data(n_days=240, n_symbols=8, seed=11):
    """随机游走 + 横截面漂移差异，保证 IC 有非零方差。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    names = [f"S{i:02d}" for i in range(n_symbols)]

    drift = np.linspace(-0.0005, 0.0008, n_symbols)
    rets = rng.normal(drift, 0.016, size=(n_days, n_symbols))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)),
                         index=dates, columns=names)
    return close


# ===================================================================
# 1. Newey-West
# ===================================================================

def test_newwey_west():
    print("\n── Newey-West t ──")

    # lag=0 退化为普通 t（总体方差口径，和 ddof=1 只差 sqrt((n-1)/n)）
    rng = np.random.default_rng(3)
    x = rng.normal(0.05, 1.0, 400)
    naive = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    check(abs(newey_west_tstat(x, 0) - naive) < 1e-12,
          "lag=0 equals the plain t statistic")

    # 正自相关序列：NW 的方差更大，t 应该收缩
    eps = rng.normal(0.0, 1.0, 800)
    ar = np.zeros_like(eps)
    for i in range(1, len(ar)):
        ar[i] = 0.85 * ar[i - 1] + eps[i]
    ar = ar + 0.03                       # 给一个非零均值
    t0 = abs(newey_west_tstat(ar, 0))
    t5 = abs(newey_west_tstat(ar, 5))
    check(t5 < t0, "positive autocorrelation shrinks |t|")

    check(newey_west_tstat([1.0, 1.0, 1.0], 2) == 0.0,
          "constant series has zero variance → t=0")
    check(newey_west_tstat([0.1], 0) == 0.0, "one sample → t=0")
    check(newey_west_tstat([], 0) == 0.0, "empty → t=0")

    # NaN 被剔除
    check(abs(newey_west_tstat([0.1, np.nan, 0.2, 0.3], 1) -
             newey_west_tstat([0.1, 0.2, 0.3], 1)) < 1e-12,
          "NaN samples are dropped")

    try:
        newey_west_tstat(x, -1)
        check(False, "negative lags should raise")
    except ValueError:
        check(True, "negative lags raise ValueError")


# ===================================================================
# 2. 因子构造
# ===================================================================

def test_lookback_factor():
    print("\n── lookback factor ──")
    close = make_data(80, 5)

    mom = make_lookback_factor(close, 5, "momentum")
    check(mom.iloc[:5].isna().all().all(), "first L rows are NaN")
    check(np.allclose(mom.iloc[5:].values,
                      close.pct_change(5).iloc[5:].values, equal_nan=True),
          "momentum = close.pct_change(L)")

    rev = make_lookback_factor(close, 5, "reversal")
    check(np.allclose(rev.fillna(0).values, -mom.fillna(0).values),
          "reversal is the negative of momentum")

    for bad in (0, -3):
        try:
            make_lookback_factor(close, bad)
            check(False, f"lookback={bad} should raise")
        except ValueError:
            check(True, f"lookback={bad} raises ValueError")

    try:
        make_lookback_factor(close, 5, "vol")
        check(False, "unknown kind should raise")
    except ValueError:
        check(True, "unknown kind raises ValueError")


# ===================================================================
# 3. 单格统计
# ===================================================================

def test_ic_stats():
    print("\n── single-cell stats ──")
    close = make_data()
    fwd = compute_forward_returns(close, 5)
    factor = make_lookback_factor(close, 20)

    st = ic_stats(factor, fwd, nw_lags=4)
    ref = compute_ic(factor, fwd)
    check(abs(st.mean_ic - ref.mean_ic) < 1e-12, "mean IC matches compute_ic")
    check(abs(st.icir - ref.icir) < 1e-12, "ICIR matches compute_ic")
    check(st.n_samples == ref.n_samples, "sample count matches")
    check(st.nw_t_stat != 0.0, "NW t is computed for a real sample")

    # 完美因子：因子就是未来收益本身，横截面秩相关为 1
    perfect = ic_stats(fwd, fwd, nw_lags=4)
    check(perfect.mean_ic > 0.999, "perfect factor has IC ≈ 1")

    # 横截面股票数不足（min_obs 看的是每天的股票数，不是天数）
    empty = ic_stats(close.iloc[:, :3], fwd.iloc[:, :3], min_obs=5)
    check(empty.n_samples == 0 and empty.mean_ic == 0.0,
          "too few symbols → empty stats")
    check(empty.nw_t_stat == 0.0, "empty stats → NW t = 0")


# ===================================================================
# 4. 两个方向的扫描
# ===================================================================

def test_profiles():
    print("\n── horizon / lookback profiles ──")
    close = make_data()

    hp = analyze_horizons(close, horizons=[1, 5, 20])
    check(set(hp) == {"momentum_20", "reversal_5", "vol_20"},
          "horizon profile covers the three default factors")
    check(set(hp["momentum_20"]) == {1, 5, 20},
          "horizon keys are preserved")

    # 和单独调用逐格一致
    fwd5 = compute_forward_returns(close, 5)
    mom20 = close.pct_change(20)
    ref = compute_ic(mom20, fwd5)
    check(abs(hp["momentum_20"][5].mean_ic - ref.mean_ic) < 1e-12,
          "momentum_20@5d matches a direct computation")

    try:
        analyze_horizons(close, horizons=[5], factors=["nope"])
        check(False, "unknown factor should raise")
    except ValueError:
        check(True, "unknown factor raises ValueError")

    lp = analyze_lookbacks(close, lookbacks=[5, 10, 20, 20], horizon=5)
    check(list(lp) == [5, 10, 20], "duplicate lookbacks are removed")
    check(all(v.n_samples > 0 for v in lp.values()),
          "every lookback produces samples")

    try:
        analyze_lookbacks(close, lookbacks=[0])
        check(False, "lookback=0 should raise")
    except ValueError:
        check(True, "lookback=0 raises ValueError")

    try:
        analyze_lookbacks(close, lookbacks=[5], horizon=0)
        check(False, "horizon=0 should raise")
    except ValueError:
        check(True, "horizon=0 raises ValueError")


# ===================================================================
# 5. 交叉网格
# ===================================================================

def test_grid():
    print("\n── lookback × horizon grid ──")
    close = make_data()
    lbs = [5, 20]
    hs = [1, 10]
    grid = analyze_grid(close, lookbacks=lbs, horizons=hs)

    check(set(grid) == {(lb, h) for lb in lbs for h in hs},
          "grid contains every combination")

    # 同一格和固定方向扫描一致
    lp = analyze_lookbacks(close, lookbacks=lbs, horizon=10)
    for lb in lbs:
        check(abs(grid[(lb, 10)].mean_ic - lp[lb].mean_ic) < 1e-12,
              f"grid ({lb},10) matches lookback scan")

    hp = analyze_horizons(close, horizons=hs)
    for h in hs:
        check(abs(grid[(20, h)].mean_ic - hp["momentum_20"][h].mean_ic) < 1e-12,
              f"grid (20,{h}) matches momentum_20 profile")

    mom10 = analyze_grid(close, lookbacks=[10], horizons=[5], kind="momentum")
    rev10 = analyze_grid(close, lookbacks=[10], horizons=[5], kind="reversal")
    check(abs(rev10[(10, 5)].mean_ic + mom10[(10, 5)].mean_ic) < 1e-12,
          "reversal flips the sign of momentum IC")


# ===================================================================
# 6. 宽表整理
# ===================================================================

def test_tables():
    print("\n── stats tables ──")
    nested = {
        "a": {1: ICStats(0.01, 0.1, 0.1, 1.0, 0.8, 0.5, 100),
              5: ICStats(0.02, 0.1, 0.2, 2.0, 1.5, 0.6, 90)},
        "b": {1: ICStats(-0.01, 0.1, -0.1, -1.0, -0.7, 0.4, 80)},
    }
    t = stats_table(nested, ["a", "b"], [1, 5], "mean_ic")
    check(abs(t.loc["a", 1] - 0.01) < 1e-12, "nested dict → table")
    check(np.isnan(t.loc["b", 5]), "missing cell becomes NaN")

    flat = {(0, 0): ICStats(0.03, 0.1, 0.3, 3.0, 2.0, 0.7, 50)}
    t2 = stats_table(flat, [0], [0], "nw_t_stat")
    check(abs(t2.iloc[0, 0] - 2.0) < 1e-12, "flat dict + field selection")

    t3 = stats_table(nested, ["a"], [1], "n_samples")
    check(int(t3.iloc[0, 0]) == 100, "n_samples field")


# ===================================================================
# 7. 分段稳定性
# ===================================================================

def test_stability():
    print("\n── segment stability ──")
    close = make_data()
    fwd = compute_forward_returns(close, 5)

    segs = subperiod_ic(fwd, fwd, n_segments=4)
    check(len(segs) == 4, "four segments returned")
    check(all(v > 0.999 for v in segs),
          "perfect factor is stable in every segment")

    try:
        subperiod_ic(fwd, fwd, n_segments=1)
        check(False, "n_segments=1 should raise")
    except ValueError:
        check(True, "n_segments=1 raises ValueError")

    check(sign_consistency([0.1, 0.2, 0.05, 0.3], 0.1) == 1.0,
          "all positive segments → 100%")
    check(sign_consistency([0.1, -0.1], 0.05) == 0.5,
          "half positive → 50%")
    check(sign_consistency([0.1, 0.2], 0.0) == 0.0,
          "zero full mean → 0%")
    check(sign_consistency([np.nan, 0.1], 0.1) == 1.0,
          "NaN segments are ignored")


# ===================================================================
# 8. 无未来信息
# ===================================================================

def test_no_lookahead():
    print("\n── no lookahead ──")
    close = make_data()
    a = make_lookback_factor(close, 20)
    b = make_lookback_factor(close.iloc[:150], 20)

    fa = compute_ic(a, compute_forward_returns(close, 5))
    fb = compute_ic(b, compute_forward_returns(close.iloc[:150], 5))

    common = fa.ic_series.index.intersection(fb.ic_series.index)
    check(len(common) > 50, "enough common dates to compare")
    check(np.allclose(fa.ic_series[common].values, fb.ic_series[common].values),
          "truncating the tail does not change earlier ICs")
    check(len(fb.ic_series) < len(fa.ic_series),
          "truncated sample simply has fewer dates")


# ===================================================================
# 9. 报告
# ===================================================================

def test_report():
    print("\n── report ──")
    close = make_data()
    hp = analyze_horizons(close, horizons=[1, 5])
    grid = analyze_grid(close, lookbacks=[5, 20], horizons=[1, 5])
    stability = {
        "momentum_20": {"full": 0.01, "segments": [0.02, 0.01, 0.0, -0.01],
                        "consistency": 0.5},
    }
    text = format_sensitivity_report(hp, grid, [5, 20], [1, 5], stability)

    for needle in ("IC 期限敏感性", "因子 × 评估期限", "信号期限 × 评估期限",
                   "Newey-West", "分段稳定性", "momentum_20"):
        check(needle in text, f"report contains '{needle}'")


def main():
    print("=" * 62)
    print("  IC sensitivity tests")
    print("=" * 62)

    test_newwey_west()
    test_lookback_factor()
    test_ic_stats()
    test_profiles()
    test_grid()
    test_tables()
    test_stability()
    test_no_lookahead()
    test_report()

    print(f"\n  Result: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
