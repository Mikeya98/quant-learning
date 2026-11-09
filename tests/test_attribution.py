"""
收益归因测试。

不依赖 pytest，直接:

    python tests/test_attribution.py

用合成价格和权重检查三件事：
    1. 每日贡献 = 前一日权重 × 当日涨幅（不能用当天权重）
    2. 个券贡献之和能对上组合收益，残差只该来自费用/成交时点
    3. Brinson 三项相加等于组合收益减基准收益（恒等式）
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.analysis.attribution import (
    benchmark_weights,
    brinson_attribution,
    contribution_summary,
    daily_contributions,
)

PASS = 0
FAIL = 0


def check(cond, name):
    global PASS, FAIL
    if cond:
        print(f"  [OK]   {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}")
        FAIL += 1


def make_data():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    prices = pd.DataFrame({
        "A": [10.0, 11.0, 12.0, 12.0, 13.0, 14.0],
        "B": [20.0, 20.0, 21.0, 22.0, 21.0, 22.0],
    }, index=dates)
    weights = pd.DataFrame({
        "date": dates,
        "A": [0.5, 0.6, 0.6, 0.3, 0.3, 0.5],
        "B": [0.5, 0.4, 0.4, 0.7, 0.7, 0.5],
    })
    return dates, prices, weights


def test_daily_contributions():
    print("-- 每日贡献 --")
    dates, prices, weights = make_data()
    daily = daily_contributions(prices, weights)

    rets = prices.pct_change().fillna(0.0)
    w = weights.set_index("date")

    # 第 2 天（index=2）：用第 1 天收盘权重 0.6/0.4
    expect = 0.4 * rets.loc[dates[2], "B"] + 0.6 * rets.loc[dates[2], "A"]
    check(abs(daily.loc[dates[2], "market"] - expect) < 1e-12,
          "day 2 contribution uses day 1 weights")
    check(abs(daily.loc[dates[2], "A"] -
              0.6 * rets.loc[dates[2], "A"]) < 1e-12,
          "symbol contribution = prev weight * return")

    # 第一天没有前一日权重，贡献为 0
    check(daily.loc[dates[0], "market"] == 0.0, "first day has no contribution")

    # 改当天的权重，不影响当天的贡献（只影响下一天）
    w2 = weights.copy()
    w2.loc[2, "A"] = 0.9
    w2.loc[2, "B"] = 0.1
    daily2 = daily_contributions(prices, w2)
    check(abs(daily2.loc[dates[2], "market"] -
              daily.loc[dates[2], "market"]) < 1e-12,
          "changing same-day weights does not leak into same-day return")
    check(abs(daily2.loc[dates[3], "market"] -
              daily.loc[dates[3], "market"]) > 1e-6,
          "changing day 2 weights changes day 3 contribution")

    # 用无费用净值构造 equity，残差应为 0
    equity = [100000.0]
    for x in daily["market"].iloc[1:]:
        equity.append(equity[-1] * (1 + x))
    eqdf = pd.DataFrame({"date": dates, "equity": equity})
    daily3 = daily_contributions(prices, weights, eqdf)
    check(abs(daily3["residual"]).max() < 1e-10,
          "no-fee equity: residual is zero")

    # 加一笔费用，残差应该为负
    equity_fee = equity.copy()
    equity_fee[3] -= 100.0
    eqdf2 = pd.DataFrame({"date": dates, "equity": equity_fee})
    daily4 = daily_contributions(prices, weights, eqdf2)
    check(daily4.loc[dates[3], "residual"] < 0,
          "fee day produces negative residual")


def test_summary():
    print("-- 个券汇总 --")
    dates, prices, weights = make_data()
    daily = daily_contributions(prices, weights)
    s = contribution_summary(daily, weights)

    check(abs(s["total"].sum() - daily["market"].sum()) < 1e-12,
          "summary totals sum to market contribution")
    check(abs(s["share_pct"].sum() - 100.0) < 1e-9,
          "shares add up to 100%")
    a = s[s["symbol"] == "A"].iloc[0]
    check(a["active_days"] == 5, "A held for 5 days (day 1 has no prev weight)")
    check(s.iloc[0]["total"] >= s.iloc[-1]["total"],
          "summary sorted by contribution descending")


def test_benchmark_weights():
    print("-- 基准权重 --")
    dates, prices, weights = make_data()
    bw = benchmark_weights(prices)
    check(np.allclose(bw.sum(axis=1), 1.0), "benchmark weights sum to 1")
    # 第一天等权
    check(abs(bw.iloc[0]["A"] - 0.5) < 1e-12 and
          abs(bw.iloc[0]["B"] - 0.5) < 1e-12, "first day is equal weight")
    # 涨得多的权重变大
    check(bw.iloc[-1]["A"] > 0.5, "winner drifts to larger weight")


def test_brinson_identity():
    print("-- Brinson 恒等式 --")
    dates, prices, weights = make_data()
    industry = pd.Series({"A": "科技", "B": "消费"})

    br = brinson_attribution(prices, weights, industry)
    a = br.attrs
    check(abs(a["identity_error"]) < 1e-9,
          f"allocation+selection+interaction = excess "
          f"(err={a['identity_error']:.2e})")
    check(abs(a["excess"] - (a["portfolio_return"] - a["benchmark_return"])) < 1e-12,
          "excess = portfolio - benchmark")
    check(abs(br["total"].sum() - a["excess"]) < 1e-9,
          "row totals sum to excess")

    # 组合等于基准 → 三项都应该接近 0
    br2 = brinson_attribution(prices, weights, industry,
                              benchmark=weights.set_index("date"))
    check(abs(br2["allocation"]).max() < 1e-12 and
          abs(br2["selection"]).max() < 1e-12 and
          abs(br2["interaction"]).max() < 1e-12,
          "portfolio == benchmark: all effects zero")

    # 有现金时恒等式也要成立
    cash_w = weights.copy()
    cash_w["A"] = cash_w["A"] * 0.6
    cash_w["B"] = cash_w["B"] * 0.6
    br3 = brinson_attribution(prices, cash_w, industry)
    check(abs(br3.attrs["identity_error"]) < 1e-9,
          "cash position keeps the identity intact")

    # 没给行业表：只有一个行业，配置项应为 0
    br4 = brinson_attribution(prices, weights)
    check(abs(br4["allocation"]).max() < 1e-12,
          "single industry: allocation effect is zero")
    check(abs(br4["total"].sum() - br4.attrs["excess"]) < 1e-9,
          "single industry total still equals excess")

    # 行业表里没有的代码落到"未分类"
    br5 = brinson_attribution(prices, weights, pd.Series({"A": "科技"}))
    check("未分类" in br5.index, "missing industry falls back to 未分类")


def main():
    print("=" * 58)
    print("  收益归因测试")
    print("=" * 58)
    test_daily_contributions()
    test_summary()
    test_benchmark_weights()
    test_brinson_identity()
    print()
    print("=" * 58)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 58)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
