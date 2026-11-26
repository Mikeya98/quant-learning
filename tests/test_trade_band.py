"""
调仓缓冲带测试。

    python tests/test_trade_band.py

覆盖：
    1. 持仓状态机 —— 掉出缓冲带才换、没掉出就不换
    2. 持续因子 —— 之后各期换手为 0
    3. 随机因子 —— band 越大换手越低
    4. 组合模拟 —— 每期收益等于持仓股票的前向收益均值
    5. 对照表 —— 换手/成本随 band 下降，净收益 = 毛收益 - 成本
    6. 文本报告
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.analysis.trade_band import (
    band_comparison,
    banded_holdings,
    banded_portfolio,
    format_band_report,
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


def make_close(n_days=260, n_stocks=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"{600000 + i:06d}" for i in range(n_stocks)]
    rets = pd.DataFrame(
        rng.normal(0, 0.015, (n_days, n_stocks)),
        index=dates, columns=cols,
    )
    return (1.0 + rets).cumprod()


def main():
    # ---------- 1. 状态机 ----------
    print("── 持仓状态机 ──")
    dates = pd.bdate_range("2021-01-01", periods=3)
    cols = ["A", "B", "C", "D", "E"]

    # 第 1 天 A 最强；第 2 天 A 掉到第 2（band=2 时保留）；
    # 第 3 天 A 掉到第 5（必须换掉）
    f = pd.DataFrame(
        [
            [5, 4, 3, 2, 1],
            [4, 5, 3, 2, 1],
            [1, 5, 4, 3, 2],
        ],
        index=dates, columns=cols, dtype=float,
    )

    rows = banded_holdings(f, holding_days=1, top_frac=0.2, band=2.0)
    # top_frac=0.2, 5 只 → n=1；band=2 → keep_n=2
    check(rows[0][1] == {"A"} and rows[0][2] == 1.0,
          "first period builds the top-1 position")
    check(rows[1][1] == {"A"} and rows[1][2] == 0.0,
          "A stays while still inside the top 2 (no trade)")
    check(rows[2][1] == {"B"} and rows[2][2] == 1.0,
          "A drops out of the top 2 and is replaced by B")

    rows_nb = banded_holdings(f, holding_days=1, top_frac=0.2, band=1.0)
    check(rows_nb[1][1] == {"B"} and rows_nb[1][2] == 1.0,
          "without a band, A is sold as soon as it slips to rank 2")

    # ---------- 2. 持续因子 ----------
    print("── 持续因子 ──")
    close = make_close()
    n_days, n_stocks = close.shape
    persistent = pd.DataFrame(
        np.tile(np.arange(n_stocks, dtype=float), (n_days, 1)),
        index=close.index, columns=close.columns,
    )
    rows = banded_holdings(persistent, holding_days=5, top_frac=0.2, band=1.5)
    check(all(r[2] == 0.0 for r in rows[1:]),
          "a persistent factor never trades after the first period")

    # ---------- 3. 随机因子：band 越大换手越低 ----------
    print("── 随机因子 ──")
    rng = np.random.default_rng(3)
    noise = pd.DataFrame(
        rng.normal(size=(n_days, n_stocks)),
        index=close.index, columns=close.columns,
    )
    to_none = np.mean([r[2] for r in banded_holdings(
        noise, holding_days=5, top_frac=0.2, band=1.0)[1:]])
    to_wide = np.mean([r[2] for r in banded_holdings(
        noise, holding_days=5, top_frac=0.2, band=2.0)[1:]])
    check(to_wide < to_none,
          f"wider band lowers turnover ({to_none:.1%} → {to_wide:.1%})")

    # ---------- 4. 组合模拟 ----------
    print("── 组合模拟 ──")
    d = pd.bdate_range("2021-01-01", periods=4)
    px = pd.DataFrame(
        [[10.0, 10.0], [11.0, 10.0], [12.1, 10.0], [13.31, 10.0]],
        index=d, columns=["A", "B"],
    )
    fac = pd.DataFrame(
        [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0]],
        index=d, columns=["A", "B"],
    )
    pf = banded_portfolio(fac, px, holding_days=1, top_frac=0.5,
                          band=1.0, cost_bps=0.0)
    # 每期做多 A（因子高）、做空 B（收益 0）→ ls_ret 就是 A 的日收益
    check(len(pf) == 3, "one row per rebalance with a forward return")
    check(np.allclose(pf["ls_ret"].values, [0.10, 0.10, 0.10], atol=1e-9),
          "long-short return equals the held stock's forward return")
    check(np.allclose(pf["turnover"].values, [1.0, 0.0, 0.0]),
          "turnover is 1.0 on entry and 0 afterwards")

    # ---------- 5. 对照表 ----------
    print("── 对照表 ──")
    table = band_comparison(
        noise, close, holding_days=5, top_frac=0.2,
        bands=(1.0, 1.5, 2.0), cost_bps=15.0,
    )
    check(list(table.index) == [1.0, 1.5, 2.0], "one row per band")
    check(table.loc[1.0, "turnover"] > table.loc[2.0, "turnover"],
          "turnover falls as the band widens")
    check(table.loc[1.0, "cost_ann"] > table.loc[2.0, "cost_ann"],
          "annual cost falls as the band widens")
    check(np.allclose(table["net_ann"],
                      table["gross_ann"] - table["cost_ann"]),
          "net = gross - cost")

    # ---------- 6. 文本报告 ----------
    print("── 文本报告 ──")
    report = format_band_report({"noise": table}, holding_days=5)
    check("调仓缓冲带" in report, "report title")
    check("noise" in report, "report names the factor")
    check("净年化" in report, "report has the net column")

    print()
    print("=" * 50)
    print(f"  Trade Band Test: {PASS} passed, {FAIL} failed")
    print("=" * 50)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
