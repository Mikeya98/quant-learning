"""
交易成本拆分测试（Phase 15）。

不依赖 pytest，直接:

    python tests/test_costs.py

覆盖：
    1. Broker 每笔成交的四项成本（佣金/印花税/过户费/滑点）
    2. cost_breakdown 的总量、方向、标的汇总
    3. 空 trades / 旧格式 trades 的兼容
    4. 组合回测产出的 trades 表确实带拆分列
"""

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


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_broker_fills():
    print("\n== Broker 成本分项 ==")
    br = Broker(commission_rate=0.00025, min_commission=5.0,
                slippage=0.001, stamp_duty_rate=0.001,
                transfer_fee_rate=0.00001)

    buy = br.execute_buy(10.0, 1000)
    # 成交价 10.01, 佣金 2.5 元被最低 5 元抬高
    check(close(buy.price, 10.01), "buy exec price includes slippage")
    check(close(buy.commission, 5.0), "buy commission hits 5 yuan minimum")
    check(buy.stamp_duty == 0.0, "buy pays no stamp duty")
    check(close(buy.transfer_fee, 10.01 * 1000 * 0.00001),
          "buy transfer fee = notional * 0.001%")
    check(close(buy.explicit_cost,
                buy.commission + buy.transfer_fee),
          "buy explicit cost = commission + transfer")
    check(close(buy.slippage_cost, 0.01 * 1000),
          "buy slippage cost = (exec - ref) * shares")
    check(close(buy.net_amount, 10.01 * 1000 + buy.explicit_cost),
          "buy net amount = notional + explicit cost")

    sell = br.execute_sell(10.0, 1000)
    check(close(sell.price, 9.99), "sell exec price includes slippage")
    check(close(sell.stamp_duty, 9.99 * 1000 * 0.001),
          "sell stamp duty = notional * 0.1%")
    check(close(sell.explicit_cost,
                sell.commission + sell.stamp_duty + sell.transfer_fee),
          "sell explicit cost includes stamp duty")
    check(close(sell.net_amount,
                9.99 * 1000 - sell.explicit_cost),
          "sell net amount = notional - explicit cost")
    check(close(sell.slippage_cost, 0.01 * 1000),
          "sell slippage measured against decision price")

    # 佣金按成交额算，不是按决策价
    big = br.execute_buy(100.0, 10000)
    check(close(big.commission, 100.1 * 10000 * 0.00025),
          "commission uses execution price")
    check(big.side == OrderSide.BUY, "fill keeps side")


def make_trades():
    return pd.DataFrame([
        # 买入 1 万，佣金 5，过户费 0.1，滑点 10
        {"date": "2020-01-02", "symbol": "A", "side": "BUY",
         "shares": 1000, "price": 10.01, "ref_price": 10.0,
         "notional": 10010.0, "commission": 5.0, "stamp_duty": 0.0,
         "transfer_fee": 0.10, "slippage": 10.0, "fee": 5.10},
        # 卖出 1 万，佣金 5，印花税 10，过户费 0.1，滑点 10
        {"date": "2020-01-03", "symbol": "A", "side": "SELL",
         "shares": 1000, "price": 9.99, "ref_price": 10.0,
         "notional": 9990.0, "commission": 5.0, "stamp_duty": 10.0,
         "transfer_fee": 0.10, "slippage": 10.0, "fee": 15.10},
        # 另一只标的只买不卖
        {"date": "2020-01-03", "symbol": "B", "side": "BUY",
         "shares": 500, "price": 20.02, "ref_price": 20.0,
         "notional": 10010.0, "commission": 5.0, "stamp_duty": 0.0,
         "transfer_fee": 0.10, "slippage": 10.0, "fee": 5.10},
    ])


def test_cost_breakdown():
    print("\n== 成本汇总 ==")
    bd = cost_breakdown(make_trades())

    notional = 10010.0 + 9990.0 + 10010.0
    explicit = (5.0 + 0.10) + (5.0 + 10.0 + 0.10) + (5.0 + 0.10)
    slippage = 30.0

    check(bd["n_trades"] == 3, "trade count")
    check(close(bd["notional"], notional), "notional total")
    check(close(bd["commission"], 15.0), "commission total")
    check(close(bd["stamp_duty"], 10.0), "stamp duty total (sell only)")
    check(close(bd["transfer_fee"], 0.30), "transfer fee total")
    check(close(bd["explicit_cost"], explicit), "explicit cost total")
    check(close(bd["slippage"], slippage), "slippage total")
    check(close(bd["total_cost"], explicit + slippage), "total cost = explicit + slippage")
    check(close(bd["explicit_bps"], explicit / notional * 10000),
          "explicit cost in bps")
    check(close(bd["slippage_bps"], slippage / notional * 10000),
          "slippage in bps")

    side = bd["by_side"]
    check(close(side.loc["BUY", "commission"], 10.0), "buy commission")
    check(close(side.loc["SELL", "stamp_duty"], 10.0), "sell stamp duty")
    check(side.loc["BUY", "stamp_duty"] == 0.0, "buy side has no stamp duty")
    check(close(side["total_cost"].sum(), bd["total_cost"]),
          "by_side total matches overall")

    sym = bd["by_symbol"]
    check(close(sym.loc["A", "total_cost"], 40.20),
          "symbol A total = buy 15.10 + sell 25.10")
    check(close(sym["total_cost"].sum(), bd["total_cost"]),
          "by_symbol total matches overall")
    check(sym.index[0] == "A", "symbols sorted by cost desc")

    report = format_cost_report(bd)
    check("交易成本拆分" in report and "成交笔数" in report,
          "report renders")


def test_empty_and_legacy():
    print("\n== 空表 / 旧格式 ==")
    empty = cost_breakdown(pd.DataFrame())
    check(empty["n_trades"] == 0 and empty["total_cost"] == 0.0,
          "empty trades -> zeros")
    check("没有成交" in format_cost_report(empty),
          "empty report says no trades")

    legacy = pd.DataFrame([
        {"date": "2020-01-02", "symbol": "A", "side": "SELL",
         "shares": 100, "price": 10.0, "fee": 6.0},
    ])
    bd = cost_breakdown(legacy)
    check(bd["n_trades"] == 1, "legacy trades counted")
    check(close(bd["notional"], 1000.0), "legacy notional derived")
    check(close(bd["explicit_cost"], 6.0) and bd["slippage"] == 0.0,
          "legacy fee treated as explicit cost")


def make_data(n=60, syms=("A", "B")):
    dates = pd.bdate_range("2020-01-01", periods=n)
    return {
        s: pd.DataFrame({"date": dates,
                         "close": [10.0 * (1.001 ** i) for i in range(n)],
                         "volume": 1e6, "amount": 1e9})
        for s in syms
    }


def test_backtest_columns():
    print("\n== 回测产出的 trades 表 ==")
    data = make_data()
    sigs = {s: pd.Series(1, index=df.index) for s, df in data.items()}
    bt = MultiAssetBacktester(initial_capital=100000, rebalance_days=5)
    r = bt.run(data, sigs, alloc="equal")

    tr = r.trades_df
    required = {"commission", "stamp_duty", "transfer_fee",
                "slippage", "notional", "ref_price"}
    check(required.issubset(tr.columns), "trades table has cost columns")
    check(len(tr) > 0, "backtest produced trades")

    if len(tr):
        buys = tr[tr["side"] == "BUY"]
        sells = tr[tr["side"] == "SELL"]
        check((buys["stamp_duty"] == 0).all(), "buy rows have no stamp duty")
        check((sells["stamp_duty"] > 0).all(), "sell rows pay stamp duty")
        check((tr["slippage"] > 0).all(), "all rows carry slippage cost")
        check(((tr["fee"] - (tr["commission"] + tr["stamp_duty"]
                             + tr["transfer_fee"])).abs() <= 0.02).all(),
              "fee column = commission + stamp + transfer (rounded)")

        bd = cost_breakdown(tr)
        check(close(bd["explicit_cost"], tr["fee"].sum(), tol=0.1),
              "breakdown explicit cost matches trades fee column")
        check(close(bd["notional"], tr["notional"].sum()),
              "breakdown notional matches trades")


def main():
    test_broker_fills()
    test_cost_breakdown()
    test_empty_and_legacy()
    test_backtest_columns()

    print()
    print(f"交易成本测试: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
