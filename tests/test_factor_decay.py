"""
因子衰减与换手率测试。

    python tests/test_factor_decay.py

覆盖：
    1. IC 衰减曲线 —— 短期因子在 h=1 最强，h=20 明显变弱
    2. IC 半衰期 —— 折线插值、负 IC 返回 None
    3. 换手率 —— 持续因子为 0，随机因子接近 1-top_frac
    4. 持有期对照表 —— 成本随持有期下降、净收益 = 毛收益 - 成本
    5. 文本报告 —— 表头与每个因子的段落都在
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.analysis.factor_decay import (
    decay_curve,
    format_decay_report,
    holding_period_table,
    ic_half_life,
    top_set_turnover,
)
from src.analysis.factor_research import compute_forward_returns

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


def make_close(n_days=420, n_stocks=40, seed=0):
    """生成随机游走收盘价宽表"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"{600000 + i:06d}" for i in range(n_stocks)]
    rets = pd.DataFrame(
        rng.normal(0, 0.015, (n_days, n_stocks)),
        index=dates, columns=cols,
    )
    close = (1.0 + rets).cumprod()
    return close, rets


def main():
    close, rets = make_close()

    # ---------- 1. IC 衰减曲线 ----------
    print("── IC 衰减曲线 ──")
    fwd1 = compute_forward_returns(close, 1)
    noise = pd.DataFrame(
        np.random.default_rng(1).normal(0, 4, fwd1.shape),
        index=fwd1.index, columns=fwd1.columns,
    )
    short_factor = fwd1 * 100.0 + noise       # 只对明天有预测力

    curve = decay_curve(short_factor, close, horizons=(1, 2, 5, 10, 20))
    check(list(curve.index) == [1, 2, 5, 10, 20], "decay_curve returns one row per horizon")
    check(curve.loc[1, "mean_ic"] > 0.15, "h=1 IC is strong for a one-day factor")
    check(curve.loc[1, "mean_ic"] > curve.loc[10, "mean_ic"],
          "IC falls as the horizon lengthens")
    check(curve.loc[1, "mean_ic"] > curve.loc[20, "mean_ic"],
          "h=20 is weaker than h=1")
    check((curve["n_samples"] > 0).all(), "every horizon has samples")

    # ---------- 2. IC 半衰期 ----------
    print("── IC 半衰期 ──")
    curve2 = pd.DataFrame(
        {"mean_ic": [0.10, 0.08, 0.06, 0.04]},
        index=[1, 2, 3, 4],
    )
    hl = ic_half_life(curve2)
    check(hl is not None and abs(hl - 3.5) < 1e-9,
          "half life interpolates between 3 and 4 (0.05 target)")

    neg = pd.DataFrame({"mean_ic": [-0.02, -0.01]}, index=[1, 2])
    check(ic_half_life(neg) is None, "negative h=1 IC has no half life")

    flat = pd.DataFrame({"mean_ic": [0.10, 0.09, 0.08]}, index=[1, 2, 3])
    check(ic_half_life(flat) is None,
          "curve that never halves returns None")

    # ---------- 3. 换手率 ----------
    print("── 换手率 ──")
    n_days, n_stocks = close.shape
    dates = close.index

    # 每只股票的因子值固定不变 → 每次选的股票完全一样
    persistent = pd.DataFrame(
        np.tile(np.arange(n_stocks, dtype=float), (n_days, 1)),
        index=dates, columns=close.columns,
    )
    check(top_set_turnover(persistent, holding_days=5, top_frac=0.2) == 0.0,
          "persistent factor has zero turnover")

    rng = np.random.default_rng(2)
    pure_noise = pd.DataFrame(
        rng.normal(size=(n_days, n_stocks)),
        index=dates, columns=close.columns,
    )
    to_noise = top_set_turnover(pure_noise, holding_days=5, top_frac=0.2)
    check(0.55 < to_noise < 0.95,
          f"random factor turnover near 1-top_frac ({to_noise:.2%})")

    # 缓慢变化的因子: 持有期越长, 两次选股之间的间隔越大, 换手越高
    slow = pure_noise.ewm(alpha=0.05, adjust=False).mean()
    to_short = top_set_turnover(slow, holding_days=1, top_frac=0.2)
    to_long = top_set_turnover(slow, holding_days=20, top_frac=0.2)
    check(to_long > to_short,
          f"longer holding period raises per-rebalance turnover "
          f"({to_short:.1%} → {to_long:.1%})")

    # ---------- 4. 持有期对照表 ----------
    print("── 持有期对照表 ──")
    table = holding_period_table(
        short_factor, close, horizons=(1, 5, 20),
        top_frac=0.2, cost_bps=15.0,
    )
    check(list(table.index) == [1, 5, 20], "table has one row per horizon")
    check((table["cost_ann"] > 0).all(), "annual cost is positive")
    check(table.loc[1, "cost_ann"] > table.loc[20, "cost_ann"],
          "short holding period costs more per year")
    check(np.allclose(table["net_ann"],
                      table["gross_ann"] - table["cost_ann"]),
          "net = gross - cost")
    check(table.loc[1, "gross_ann"] > table.loc[20, "gross_ann"],
          "gross return decays with the horizon")

    # ---------- 5. 文本报告 ----------
    print("── 文本报告 ──")
    report = format_decay_report({"short_factor": table}, cost_bps=15.0)
    check("因子衰减与持有期" in report, "report title")
    check("short_factor" in report, "report names the factor")
    check("净年化" in report, "report has the net column")

    print()
    print("=" * 50)
    print(f"  Factor Decay Test: {PASS} passed, {FAIL} failed")
    print("=" * 50)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
