"""
因子研究的正确性测试。

不依赖 pytest，直接:

    python tests/test_factor_research.py

用合成数据检查:
1. 前向收益的 shift 方向没搞反，最后 N 行为 NaN（不能有未来数据）
2. 因子 = 未来收益时 IC 恰好为 1，取负号后为 -1
3. 分层收益单调，Q5 > Q1
4. 横截面 z-score 每行均值 0、标准差 1
5. 多因子等权合成后仍然是每天一个打分
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.analysis.factor_research import (
    compute_forward_returns, compute_ic, quantile_returns,
    quantile_summary, zscore_cross_section, combine_factors, make_factors,
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


def make_close(n_days=60, n_stocks=10, seed=7):
    """生成随机游走的价格宽表"""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, size=(n_days, n_stocks))
    prices = 10 * np.cumprod(1 + rets, axis=0)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_stocks)]
    return pd.DataFrame(prices, index=dates, columns=cols)


def main():
    print("=" * 62)
    print("  Factor Research Tests")
    print("=" * 62)

    # ---------- 1. 前向收益 ----------
    print("\n-- forward returns --")
    close = make_close()
    fwd = compute_forward_returns(close, periods=5)

    manual = close.iloc[5, 0] / close.iloc[0, 0] - 1
    check(abs(fwd.iloc[0, 0] - manual) < 1e-12, "fwd[0] = close[5]/close[0]-1")
    check(fwd.iloc[-5:].isna().all().all(), "last 5 rows are NaN (no future data)")
    check(fwd.iloc[:5].notna().all().all(), "first rows are valid")

    # ---------- 2. IC 的极端情况 ----------
    print("\n-- IC --")
    # 因子就是未来收益 → 每天完全正相关
    perfect = compute_ic(fwd, fwd, method="spearman")
    check(abs(perfect.mean_ic - 1.0) < 1e-9, "factor = fwd return → IC = 1")

    neg = compute_ic(-fwd, fwd, method="spearman")
    check(abs(neg.mean_ic + 1.0) < 1e-9, "negated factor → IC = -1")

    # 随机因子 → IC 接近 0
    rng = np.random.default_rng(1)
    noise = pd.DataFrame(rng.normal(size=close.shape),
                         index=close.index, columns=close.columns)
    rnd = compute_ic(noise, fwd, method="spearman")
    check(abs(rnd.mean_ic) < 0.1, "random factor → |IC| small")

    # 样本不足时跳过
    tiny = compute_ic(fwd.iloc[:, :3], fwd.iloc[:, :3], min_obs=5)
    check(tiny.n_samples == 0, "cross-section < min_obs is skipped")

    # ---------- 3. 分层收益 ----------
    print("\n-- quantile returns --")
    qret = quantile_returns(fwd, fwd, n_quantiles=5)
    means = qret.mean()
    check(list(qret.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"],
          "columns are Q1..Q5")
    check(means["Q5"] > means["Q1"], "Q5 (high factor) > Q1 (low factor)")
    check(means.is_monotonic_increasing, "quantile means are monotonic")

    # 因子取负 → 单调方向反过来
    qret_neg = quantile_returns(-fwd, fwd, n_quantiles=5)
    check(qret_neg.mean().is_monotonic_decreasing,
          "negated factor → monotonic decreasing")

    qs = quantile_summary(qret, holding_period=5)
    check("annual" in qs.columns and len(qs) == 5, "quantile summary shape")

    # ---------- 4. 横截面标准化 ----------
    print("\n-- z-score --")
    raw = pd.DataFrame({
        "a": [1.0, 10.0],
        "b": [2.0, 20.0],
        "c": [3.0, 30.0],
    }, index=pd.to_datetime(["2021-01-04", "2021-01-05"]))
    z = zscore_cross_section(raw)
    check(np.allclose(z.mean(axis=1), 0.0, atol=1e-12), "row mean = 0")
    check(np.allclose(z.std(axis=1, ddof=0), 1.0, atol=1e-12), "row std = 1")

    # 全是同一个值 → std=0，结果应当是 0 而不是 NaN/Inf
    same = pd.DataFrame({"a": [5.0], "b": [5.0], "c": [5.0]})
    zs = zscore_cross_section(same)
    check((zs.fillna(0) == 0).all().all(), "zero-variance row → 0 scores")

    # ---------- 5. 多因子合成 ----------
    print("\n-- combine factors --")
    f1 = fwd
    f2 = -fwd
    combined = combine_factors({"a": f1, "b": f2})
    check(combined.shape == fwd.shape, "combined shape matches input")
    # 互补的两个因子等权相加 → 每天得分全为 0
    check(np.allclose(combined.fillna(0).values, 0.0, atol=1e-9),
          "factor + its negation → zero score")

    only_a = combine_factors({"a": f1, "b": f2}, weights={"a": 1.0, "b": 0.0})
    zf = zscore_cross_section(f1)
    check(np.allclose(only_a.fillna(0).values, zf.fillna(0).values, atol=1e-9),
          "weight 0 ignores the other factor")

    # ---------- 6. 内置因子库 ----------
    print("\n-- factor library --")
    volume = close * 1000
    factors = make_factors(close, volume, turnover=volume / 1e6)
    for name in ("momentum_20", "reversal_5", "vol_20", "vol_ratio", "turnover_20"):
        check(name in factors, f"factor {name} exists")
    check(factors["reversal_5"].equals(-close.pct_change(5)),
          "reversal_5 = -5d return")

    print("\n" + "=" * 62)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 62)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
