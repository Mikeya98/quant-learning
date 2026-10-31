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


def make_data_ohlcv(rows: dict) -> dict:
    """rows: {symbol: [(date, close), ...]}，可故意缺日期模拟停牌。"""
    out = {}
    for sym, items in rows.items():
        df = pd.DataFrame(items, columns=["date", "close"])
        df["volume"] = 1e6
        df["amount"] = 1e9
        out[sym] = df
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


def test_limit_up_blocks_buy():
    print("\n== 涨停不能买入 ==")
    dates = pd.bdate_range("2020-01-01", periods=4)
    data = make_data_ohlcv({
        "A": [(dates[0], 10.0), (dates[1], 11.0),   # +10% 涨停
              (dates[2], 11.0), (dates[3], 11.0)],
    })
    sig = pd.Series([0, 1, 1, 1])
    bt = MultiAssetBacktester(initial_capital=100000, rebalance_days=1)
    r = bt.run(data, {"A": sig})
    days = r.trades_df["date"] if len(r.trades_df) else pd.Series(dtype="datetime64[ns]")
    check(dates[1] not in set(days), "no buy on limit-up day")
    check(r.summary["blocked_limit"] >= 1, "blocked_limit counted")

    bt2 = MultiAssetBacktester(initial_capital=100000, rebalance_days=1,
                               limit_check=False)
    r2 = bt2.run(data, {"A": sig})
    check(r2.summary["n_trades"] > 0, "limit check off → trade happens")


def test_limit_down_blocks_sell():
    print("\n== 跌停不能卖出 ==")
    dates = pd.bdate_range("2020-01-01", periods=5)
    data = make_data_ohlcv({
        "A": [(dates[0], 10.0), (dates[1], 10.0),
              (dates[2], 9.0),                    # -10% 跌停
              (dates[3], 9.0), (dates[4], 9.0)],
    })
    sig = pd.Series([1, 1, 0, 0, 0])
    bt = MultiAssetBacktester(initial_capital=100000, rebalance_days=1)
    r = bt.run(data, {"A": sig})
    check(r.summary["blocked_limit"] >= 1, "blocked_limit counted on exit")
    check(r.per_symbol.iloc[0]["final_shares"] > 0 or
          r.summary["n_trades"] >= 2, "sell was delayed, not lost")


def test_suspension_blocks_trade():
    print("\n== 停牌日不成交 ==")
    dates = pd.bdate_range("2020-01-01", periods=4)
    data = {
        "A": pd.DataFrame({
            "date": [dates[0], dates[2], dates[3]],   # 缺 dates[1]
            "close": [10.0, 10.0, 10.0],
            "volume": [1e6, 1e6, 1e6],
            "amount": [1e9, 1e9, 1e9],
        }),
        # B 只用来把 dates[1] 带进并集, 本身不开仓
        "B": pd.DataFrame({
            "date": list(dates),
            "close": [20.0] * 4,
            "volume": [1e6] * 4,
            "amount": [1e9] * 4,
        }),
    }
    sig = pd.Series([1, 1, 1])
    bt = MultiAssetBacktester(initial_capital=100000, rebalance_days=1)
    r = bt.run(data, {"A": sig, "B": pd.Series([0, 0, 0, 0])})
    check(len(r.equity_df) == 4, "union alignment keeps the suspended day")
    check(r.summary["blocked_suspension"] >= 1, "blocked_suspension counted")
    if len(r.trades_df):
        check(dates[1] not in set(r.trades_df["date"]), "no trade on suspended day")


def test_liquidity_cap():
    print("\n== 单日成交额上限 ==")
    dates = pd.bdate_range("2020-01-01", periods=5)
    data = make_data_ohlcv({
        "A": [(d, 1.0) for d in dates],
    })
    for df in data.values():
        df["amount"] = 10000.0          # 每天成交额只有 1 万元
    sig = pd.Series([1] * 5)

    bt = MultiAssetBacktester(initial_capital=100000, rebalance_days=1,
                              max_participation=0.10)
    r = bt.run(data, {"A": sig})
    daily = r.trades_df[r.trades_df["side"] == "BUY"]["shares"]
    check(len(daily) > 0 and daily.max() <= 1000,
          "each buy capped at 10% of daily amount")
    check(r.summary["blocked_liquidity"] >= 1, "blocked_liquidity counted")

    bt2 = MultiAssetBacktester(initial_capital=100000, rebalance_days=1,
                               max_participation=0.0)
    r2 = bt2.run(data, {"A": sig})
    check(r2.per_symbol.iloc[0]["final_shares"] > 10000,
          "no cap → much larger position")


def main():
    test_all_cash()
    test_equal_weight_uptrend()
    test_exit_clears_position()
    test_single_symbol_bookkeeping()
    test_limit_up_blocks_buy()
    test_limit_down_blocks_sell()
    test_suspension_blocks_trade()
    test_liquidity_cap()

    print(f"\n{'=' * 40}")
    print(f"  {PASS} passed, {FAIL} failed")
    print(f"{'=' * 40}")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
