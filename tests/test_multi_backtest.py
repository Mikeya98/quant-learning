"""
多标的组合回测的记账不变量测试。

不依赖 pytest，直接:

    python tests/test_multi_backtest.py

合成数据，检查几件事:
1. 没有信号 → 全是现金, 净值不变
2. 等权持有两个上涨标的 → 权重和 <= 1, 现金 >= 0, 净值大于初始
3. 信号退出 → 对应标的仓位清零 (不能因为低于阈值留下尾巴)
4. 两个相同价格路径 → 等权基准 = 初始资金 × 价格涨幅
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.engine import MultiAssetBacktester

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


def make_data(prices: dict) -> dict:
    dates = pd.bdate_range("2020-01-01", periods=len(next(iter(prices.values()))))
    out = {}
    for sym, series in prices.items():
        out[sym] = pd.DataFrame({"date": dates, "close": series})
    return out


def zero_signals(data):
    return {s: pd.Series(0, index=df.index) for s, df in data.items()}


def test_all_cash():
    print("\n== 无信号: 全现金 ==")
    data = make_data({
        "A": [10.0 + i * 0.1 for i in range(30)],
        "B": [20.0 + i * 0.2 for i in range(30)],
    })
    bt = MultiAssetBacktester(initial_capital=100000, rebalance_days=1)
    r = bt.run(data, zero_signals(data))

    check(r.summary["final_equity"] == 100000, "final equity == 100000")
    check(r.summary["n_trades"] == 0, "no trades")
    check((r.equity_df["cash"] == 100000).all(), "cash unchanged")
    check((r.weights_df.drop(columns=["date"]).sum(axis=1) == 0).all(),
          "all weights are zero")


def test_equal_weight_uptrend():
    print("\n== 等权持有两个上涨标的 ==")
    n = 60
    data = make_data({
        "A": [10.0 * (1.001 ** i) for i in range(n)],
        "B": [20.0 * (1.001 ** i) for i in range(n)],
    })
    sigs = {s: pd.Series(1, index=df.index) for s, df in data.items()}
    bt = MultiAssetBacktester(initial_capital=100000, rebalance_days=5)
    r = bt.run(data, sigs, alloc="equal")

    wsum = r.weights_df.drop(columns=["date"]).sum(axis=1)
    check((wsum <= 1.0 + 1e-6).all(), "weight sum <= 1 every day")
    check((wsum > 0.9).max(), "portfolio is invested")
    check((r.equity_df["cash"] >= -1e-6).all(), "cash never negative")
    check(r.summary["final_equity"] > 100000, "equity grew")
    # 两个标的走势相同 → 基准应该和价格涨幅一致
    expected = 100000 * (1.001 ** (n - 1))
    check(abs(r.equity_df["benchmark"].iloc[-1] - expected) < 1.0,
          "benchmark matches price path")


def test_exit_clears_position():
    print("\n== 信号退出: 仓位清零 ==")
    n = 40
    data = make_data({
        "A": [10.0] * n,          # 价格不动, 避免涨跌干扰
        "B": [20.0] * n,
    })
    sig_a = pd.Series([1] * 10 + [0] * (n - 10))
    sig_b = pd.Series([1] * n)
    bt = MultiAssetBacktester(initial_capital=100000, rebalance_days=5)
    r = bt.run(data, {"A": sig_a, "B": sig_b}, alloc="equal")

    dates = r.weights_df["date"]
    after = r.weights_df[dates >= dates.iloc[10]]
    check((after["A"] == 0).all(), "A weight is zero after signal exits")
    check((r.equity_df["cash"] >= -1e-6).all(), "cash never negative")


def test_single_symbol_bookkeeping():
    print("\n== 单标的记账 ==")
    n = 30
    data = make_data({"A": [10.0 + i for i in range(n)]})
    sigs = {"A": pd.Series([1] * 15 + [0] * (n - 15))}
    bt = MultiAssetBacktester(initial_capital=50000, rebalance_days=1)
    r = bt.run(data, sigs, alloc="equal")

    # 期初买入, 第 15 天卖出 → 期末应为纯现金
    check(r.per_symbol.iloc[0]["final_shares"] == 0, "position closed at end")
    check(abs(r.equity_df["position_value"].iloc[-1]) < 1e-6,
          "position value is zero at end")
    check(r.summary["n_trades"] >= 2, "at least one buy and one sell")


def main():
    test_all_cash()
    test_equal_weight_uptrend()
    test_exit_clears_position()
    test_single_symbol_bookkeeping()

    print(f"\n{'=' * 40}")
    print(f"  {PASS} passed, {FAIL} failed")
    print(f"{'=' * 40}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
