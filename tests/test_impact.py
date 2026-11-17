"""
平方根冲击成本模型测试（Phase 16）。

不依赖 pytest，直接:

    venv/Scripts/python.exe tests/test_impact.py

覆盖：
    1. fixed 模型行为不变（兼容 Phase 15）
    2. sqrt 模型的参与率、冲击公式、上下限
    3. 成交额缺失时退化为固定滑点
    4. calc_shares 把冲击算进预算，不会超资金
    5. 组合回测的 trades 表带 participation / impact_bps
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.analysis.costs import cost_breakdown, format_cost_report
from src.engine import MultiAssetBacktester
from src.engine.broker import Broker, OrderSide

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


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_fixed_model_unchanged():
    print("\n== fixed 模型兼容 ==")
    br = Broker(slippage=0.001)
    fill = br.execute_buy(10.0, 1000, daily_amount=1e6)
    check(close(fill.price, 10.01), "fixed: buy price = 10.01")
    check(close(fill.participation, 1000 * 10.0 / 1e6),
          "fixed: participation still recorded")
    check(fill.impact_rate == 0.0, "fixed: no impact")

    fill = br.execute_sell(10.0, 1000, daily_amount=1e6)
    check(close(fill.price, 9.99), "fixed: sell price = 9.99")


def test_sqrt_formula():
    print("\n== 平方根公式 ==")
    br = Broker(slippage=0.001, slippage_model="sqrt",
                impact_coeff=0.01, max_impact=0.05)

    # 10000 股 × 10 元 = 10 万元, 当日成交额 100 万 → 参与率 10%
    fill = br.execute_buy(10.0, 10000, daily_amount=1e6)
    expected_part = 0.10
    expected_impact = 0.01 * math.sqrt(expected_part)
    check(close(fill.participation, expected_part),
          "participation = notional / daily amount")
    check(close(fill.impact_rate, expected_impact),
          "impact = coeff * sqrt(participation)")
    check(close(fill.price, 10.0 * (1 + 0.001 + expected_impact)),
          "buy price includes fixed slippage + impact")

    # 卖出方向镜像
    fill = br.execute_sell(10.0, 10000, daily_amount=1e6)
    check(close(fill.price, 10.0 * (1 - 0.001 - expected_impact)),
          "sell price subtracts fixed slippage + impact")

    # 参与率越高, 冲击越大, 但受上限约束
    small = br.execute_buy(10.0, 1000, daily_amount=1e6)
    large = br.execute_buy(10.0, 50000, daily_amount=1e6)
    check(large.impact_rate > small.impact_rate,
          "impact grows with participation")

    huge = br.execute_buy(10.0, 100000, daily_amount=1e4)
    check(close(huge.participation, 100.0) and close(huge.impact_rate, 0.05),
          "impact capped at max_impact")

    # 成交额缺失 → 没有冲击
    fill = br.execute_buy(10.0, 10000, daily_amount=None)
    check(fill.impact_rate == 0.0 and close(fill.price, 10.01),
          "missing daily amount falls back to fixed slippage")

    fill = br.execute_buy(10.0, 10000, daily_amount=float("nan"))
    check(fill.impact_rate == 0.0, "NaN daily amount is ignored")


def test_calc_shares_budget():
    print("\n== calc_shares 预算 ==")
    br = Broker(slippage=0.001, slippage_model="sqrt",
                impact_coeff=0.01, max_impact=0.05)
    cash = 100000.0
    price = 10.0
    amount = 1e6

    shares = br.calc_shares(cash, price, 1.0, daily_amount=amount)
    fill = br.execute_buy(price, shares, daily_amount=amount)
    check(shares > 0, "shares > 0")
    check(fill.total_cost <= cash + 1e-6,
          "notional + commission <= cash with impact")
    check(fill.participation <= 0.10 + 1e-9,
          "participation <= daily cap (100k / 1M)")

    no_impact = br.calc_shares(cash, price, 1.0, daily_amount=None)
    check(shares <= no_impact, "impact model buys no more than fixed model")


def test_backtester_trades_columns():
    print("\n== 组合回测 trades 列 ==")
    n = 40
    dates = pd.bdate_range("2020-01-01", periods=n)
    data = {}
    for sym, base in (("A", 10.0), ("B", 20.0)):
        data[sym] = pd.DataFrame({
            "date": dates,
            "close": [base] * n,
            "volume": 1e6,
            "amount": 1e6,
        })
    sigs = {s: pd.Series(1, index=df.index) for s, df in data.items()}

    bt = MultiAssetBacktester(
        initial_capital=100000, rebalance_days=5,
        max_participation=0.10, slippage_model="sqrt",
        impact_coeff=0.01, max_impact=0.05,
    )
    r = bt.run(data, sigs, alloc="equal")
    t = r.trades_df

    check(len(t) > 0, "trades produced")
    check("participation" in t.columns and "impact_bps" in t.columns,
          "trades contain participation / impact_bps")
    check(t["participation"].max() <= 0.10 + 1e-9,
          "no trade above max participation")
    check((t["impact_bps"] > 0).all(),
          "every trade has positive impact under sqrt model")
    check(t["impact_bps"].max() <= 500.0 + 1e-6,
          "impact capped at 5%")

    bd = cost_breakdown(t)
    check("impact_cost" in bd, "cost report includes impact section")
    check(bd["impact_cost"] > 0, "impact cost > 0")
    check(0 <= bd["impact_share"] <= 1.0 + 1e-9,
          "impact share of slippage in [0, 1]")
    text = format_cost_report(bd)
    check("冲击成本" in text, "formatted report shows impact block")

    # fixed 模型同一份数据 → impact 列为 0
    bt2 = MultiAssetBacktester(
        initial_capital=100000, rebalance_days=5,
        max_participation=0.10, slippage_model="fixed",
    )
    r2 = bt2.run(data, sigs, alloc="equal")
    check((r2.trades_df["impact_bps"] == 0).all(),
          "fixed model leaves impact_bps = 0")
    check(r2.summary["final_equity"] >= r.summary["final_equity"] - 1e-6,
          "impact model is not more profitable than fixed")


def main():
    test_fixed_model_unchanged()
    test_sqrt_formula()
    test_calc_shares_budget()
    test_backtester_trades_columns()
    print(f"\n{'=' * 60}")
    print(f"  Result: {PASS} passed, {FAIL} failed")
    print(f"{'=' * 60}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
