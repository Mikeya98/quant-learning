"""
因子选股模块测试。

    python tests/test_factor_selection.py

覆盖三件事:
    1. 排序选股 —— 分数最高的 top_n 只被选中，调仓日之间持仓不变
    2. 未来信息 —— 截断后面的数据不会改变前面任何一天的信号
    3. 边界条件 —— 数据不足、分数并列、未知因子、停牌日 ffill
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.strategy.factor_selection import (
    composite_score,
    ic_weighted_score,
    rolling_ic_weights,
    select_top,
    signals_from_selection,
    selection_turnover,
)
from src.analysis.factor_research import (
    compute_forward_returns,
    compute_ic,
    make_factors,
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


def make_series(growth, periods=120, start=10.0):
    """每天固定涨幅的净值序列"""
    return start * np.cumprod(np.full(periods, 1.0 + growth))


def make_panel(periods=120):
    idx = pd.bdate_range("2020-01-01", periods=periods)
    return pd.DataFrame({
        "A": make_series(0.010, periods),
        "B": make_series(0.006, periods),
        "C": make_series(0.003, periods),
        "D": make_series(-0.002, periods),
        "E": make_series(0.000, periods),
    }, index=idx)


def main():
    # =================================================================
    # 1. 合成分数
    # =================================================================
    print("── composite score ──")
    close = make_panel()
    score = composite_score(close, factors=["momentum_20"])

    check(score.iloc[:20].isna().all().all(),
          "first 20 days have no momentum score")
    valid = score.iloc[20:]
    check(valid["A"].notna().all(), "scores available after warm-up")
    check(bool((valid["A"] > valid["B"]).all())
          and bool((valid["B"] > valid["C"]).all()),
          "faster-growing symbols score higher")

    try:
        composite_score(close, factors=["no_such_factor"])
        check(False, "unknown factor raises")
    except ValueError:
        check(True, "unknown factor raises")

    # =================================================================
    # 2. 选股结果
    # =================================================================
    print("\n── select top N ──")
    selected = select_top(score, top_n=2, rebalance_days=10)

    check(not selected.iloc[:20].any().any(),
          "no position before enough history")
    check(selected.iloc[20:]["A"].all() and selected.iloc[20:]["B"].all(),
          "top 2 momentum names held")
    check(not selected.iloc[20:][["C", "D", "E"]].any().any(),
          "lower-ranked names never selected")

    blocks_constant = True
    for start in range(20, len(selected) - 10, 10):
        block = selected.iloc[start:start + 10]
        sets = {frozenset(block.columns[block.loc[d].to_numpy()])
                for d in block.index}
        if len(sets) != 1:
            blocks_constant = False
            break
    check(blocks_constant, "holdings stay constant between rebalance days")

    # =================================================================
    # 3. 未来信息
    # =================================================================
    print("\n── look-ahead ──")
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=140)
    rets = rng.normal(0.0005, 0.02, size=(140, 6))
    rand_close = pd.DataFrame(10 * np.cumprod(1 + rets, axis=0),
                              index=idx, columns=list("ABCDEF"))

    full = composite_score(rand_close, factors=["momentum_20", "vol_20"])
    cut = composite_score(rand_close.iloc[:100],
                          factors=["momentum_20", "vol_20"])
    check(np.allclose(full.iloc[:100].to_numpy(dtype=float),
                      cut.to_numpy(dtype=float), equal_nan=True),
          "truncating future rows does not change past scores")

    sig_full = select_top(full, top_n=3, rebalance_days=7)
    sig_cut = select_top(cut, top_n=3, rebalance_days=7)
    check(np.array_equal(sig_full.iloc[:100].to_numpy(), sig_cut.to_numpy()),
          "truncating future rows does not change past signals")

    # =================================================================
    # 4. 边界条件
    # =================================================================
    print("\n── edge cases ──")
    periods = 40
    idx2 = pd.bdate_range("2021-01-01", periods=periods)
    flat = make_series(0.005, periods)
    tie_close = pd.DataFrame({c: flat for c in "XYZ"}, index=idx2)
    tie_score = composite_score(tie_close, factors=["momentum_20"])
    tie_sel = select_top(tie_score, top_n=2, rebalance_days=5)
    held = list(tie_sel.columns[tie_sel.iloc[-1].to_numpy()])
    check(held == ["X", "Y"], "ties are broken deterministically by column order")

    short_idx = pd.bdate_range("2022-01-01", periods=10)
    short_close = pd.DataFrame({"A": make_series(0.01, 10),
                                "B": make_series(0.02, 10)}, index=short_idx)
    short_score = composite_score(short_close, factors=["momentum_20"])
    short_sel = select_top(short_score, top_n=1, rebalance_days=3)
    check(not short_sel.any().any(), "no selection without enough history")
    check(selection_turnover(short_sel)["changed_days"] == 0,
          "turnover stats handle an empty selection")

    filtered = select_top(score, top_n=5, rebalance_days=10, min_score=100.0)
    check(not filtered.any().any(), "min_score filters every candidate")

    # =================================================================
    # 5. 信号对齐
    # =================================================================
    print("\n── signal alignment ──")
    data = {
        "A": pd.DataFrame({"date": close.index, "close": close["A"]}),
        # B 少一天数据: 信号按自己的日期重新对齐并沿用前值
        "B": pd.DataFrame({"date": close.index.delete(50),
                           "close": close["B"].drop(close.index[50])}),
    }
    sig = signals_from_selection(selected, data)

    check(len(sig["A"]) == len(close), "signal length matches symbol rows")
    check(len(sig["B"]) == len(close) - 1,
          "missing day is dropped from that symbol's signal")
    check(np.array_equal(sig["A"].to_numpy(), selected["A"].to_numpy()),
          "signal values follow selection on full data")

    b_dates = data["B"]["date"]
    expected = selected["B"].reindex(b_dates).ffill().fillna(False).astype(int)
    check(np.array_equal(sig["B"].to_numpy(), expected.to_numpy()),
          "missing day filled from the previous holding state")

    # =================================================================
    # 6. 滚动 IC 加权
    # =================================================================
    print("\n── IC weighting ──")
    rng2 = np.random.default_rng(23)
    idx3 = pd.bdate_range("2019-01-01", periods=160)
    rets2 = rng2.normal(0.0002, 0.018, size=(160, 8))
    close3 = pd.DataFrame(10 * np.cumprod(1 + rets2, axis=0),
                          index=idx3, columns=[f"S{i}" for i in range(8)])

    lib = make_factors(close3)
    fwd3 = compute_forward_returns(close3, periods=10)
    chosen = {"momentum_20": lib["momentum_20"],
              "reversal_5": lib["reversal_5"]}
    w = rolling_ic_weights(chosen, fwd3, index=close3.index,
                           ic_period=10, ic_window=30, min_ic_obs=10)
    check(list(w.columns) == ["momentum_20", "reversal_5"],
          "weight table has one column per factor")
    check(w.iloc[:20].isna().all().all(),
          "weights are NaN during warm-up (history + IC lag)")

    # 手工复算：IC 序列 -> rolling mean -> shift(ic_period)
    manual = {}
    for name, df in chosen.items():
        ic = compute_ic(df, fwd3).ic_series.reindex(close3.index)
        manual[name] = ic.rolling(30, min_periods=10).mean().shift(10)
    manual = pd.DataFrame(manual, index=close3.index)
    check(np.allclose(w.to_numpy(dtype=float), manual.to_numpy(dtype=float),
                      equal_nan=True),
          "weights equal IC rolling mean shifted by ic_period")

    score3 = ic_weighted_score(close3, factors=["momentum_20", "reversal_5"],
                               ic_period=10, ic_window=30, min_ic_obs=10)
    check(score3.iloc[:20].isna().all().all(),
          "IC-weighted score waits for weights")
    later = score3.iloc[20:]
    check(later.notna().any().any(), "IC-weighted score becomes available")

    # 截断未来数据，历史分数不变
    cut3 = ic_weighted_score(close3.iloc[:120],
                             factors=["momentum_20", "reversal_5"],
                             ic_period=10, ic_window=30, min_ic_obs=10)
    check(np.allclose(score3.iloc[:120].to_numpy(dtype=float),
                      cut3.to_numpy(dtype=float), equal_nan=True),
          "IC weighting does not look ahead")

    sel3 = select_top(score3, top_n=3, rebalance_days=10)
    check(sel3.iloc[20:].any().any(), "IC-weighted score produces holdings")

    # =================================================================
    # 7. 换手统计
    # =================================================================
    print("\n── turnover stats ──")
    stats = selection_turnover(selected)
    check(stats["n_days"] == len(selected), "n_days counted")
    check(stats["avg_names"] == 2.0, "average holding count = top_n")
    check(stats["changed_days"] == 1,
          "constant trend triggers exactly one change (build position)")
    check(stats["avg_change_pct"] > 0, "change ratio recorded")

    print(f"\n  Result: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
