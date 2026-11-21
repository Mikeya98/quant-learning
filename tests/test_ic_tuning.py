"""
IC 窗口 walk-forward 调参测试。

    python tests/test_ic_tuning.py

覆盖：
    1. 参数网格 —— 笛卡尔积、去重、非法值
    2. 时间切分 —— 训练/测试不重叠、不遗漏、expanding/rolling 两种模式
    3. IC 评估 —— 完美因子的 IC = 1，窗口切片只取指定日期
    4. 单折调参 —— 训练段选中参数，测试段 IC 与单独计算一致
    5. 无未来信息 —— 截断后面的数据不改变前面的分数
    6. 汇总/报告 —— 升降幅、胜率、参数稳定性
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itertools import product

import numpy as np
import pandas as pd

from src.analysis.factor_research import compute_forward_returns
from src.strategy.factor_selection import ic_weighted_score
from src.strategy.ic_tuning import (
    TuneRow,
    format_tuning_report,
    make_param_grid,
    score_in_window,
    summarize_tuning,
    tune_fold,
    walk_forward_splits,
    walk_forward_tune,
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


def make_data(n_days=320, n_symbols=8, seed=7):
    """随机游走 + 温和的横截面差异，保证 IC 有非零方差。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    names = [f"S{i:02d}" for i in range(n_symbols)]

    drift = np.linspace(-0.0004, 0.0006, n_symbols)
    rets = rng.normal(drift, 0.018, size=(n_days, n_symbols))
    close = pd.DataFrame(
        10.0 * np.cumprod(1.0 + rets, axis=0),
        index=dates, columns=names,
    )
    volume = pd.DataFrame(
        rng.lognormal(12.0, 0.4, size=(n_days, n_symbols)),
        index=dates, columns=names,
    )
    turnover = pd.DataFrame(
        rng.lognormal(-1.0, 0.3, size=(n_days, n_symbols)),
        index=dates, columns=names,
    )
    return close, volume, turnover


def test_grid():
    print("\n── parameter grid ──")
    grid = make_param_grid((5, 10), (20, 60))
    check(len(grid) == 4, "cartesian product size")
    check(grid[0] == {"ic_period": 5, "ic_window": 20}, "first combo")
    check(grid[-1] == {"ic_period": 10, "ic_window": 60}, "last combo")
    check({(g["ic_period"], g["ic_window"]) for g in grid} ==
          set(product((5, 10), (20, 60))), "no duplicates, all combos present")

    for bad in [((5, 5), (20,)), ((0, 5), (20,)), ((5,), (20, 0))]:
        try:
            make_param_grid(*bad)
            ok = False
        except ValueError:
            ok = True
        check(ok, f"rejects invalid grid {bad}")


def test_splits():
    print("\n── walk-forward splits ──")
    index = pd.bdate_range("2020-01-01", periods=300)

    folds = walk_forward_splits(index, n_folds=3, initial_train=0.5)
    check(len(folds) == 3, "three folds")
    check(folds[0].train_index[-1] < folds[0].test_index[0],
          "train ends before test (fold 0)")
    check(folds[-1].test_index[-1] == index[-1], "last test reaches the end")

    covered = pd.Index([]).append([f.test_index for f in folds])
    expected = index[index.get_loc(folds[0].test_index[0]):]
    check(len(covered) == len(expected) and covered.equals(expected),
          "test blocks cover the tail without gaps or overlap")

    sizes = [len(f.train_index) for f in folds]
    check(sizes == sorted(sizes) and sizes[0] < sizes[-1],
          "expanding window grows")

    rolling = walk_forward_splits(index, n_folds=3, initial_train=0.5,
                                  mode="rolling")
    rsizes = [len(f.train_index) for f in rolling]
    check(rsizes == [150, 150, 150], "rolling window keeps a fixed length")
    check(rolling[1].train_index[0] > rolling[0].train_index[0],
          "rolling window moves forward")

    for kwargs in [
        dict(n_folds=0),
        dict(n_folds=3, initial_train=1.0),
        dict(n_folds=3, mode="weird"),
        dict(n_folds=50, min_test=20),
    ]:
        try:
            walk_forward_splits(index, **kwargs)
            ok = False
        except ValueError:
            ok = True
        check(ok, f"rejects invalid split {kwargs}")


def test_ic_window():
    print("\n── IC window evaluation ──")
    close, volume, turnover = make_data()
    fwd = compute_forward_returns(close, periods=5)

    # 完美预测：分数就是未来 5 日收益 → 逐日 RankIC 恒为 1
    mean_ic, icir, n = score_in_window(fwd, close, close.index, period=5)
    check(abs(mean_ic - 1.0) < 1e-9, "perfect score gives IC = 1")
    check(n > 0, "valid IC samples counted")

    # 反着用 → IC = -1
    mean_ic, _, _ = score_in_window(-fwd, close, close.index, period=5)
    check(abs(mean_ic + 1.0) < 1e-9, "negated score gives IC = -1")

    part = close.index[100:140]
    m_full, _, n_full = score_in_window(fwd, close, close.index, period=5)
    m_part, _, n_part = score_in_window(fwd, close, part, period=5)
    check(n_part <= 40, "window limits the day count")
    check(np.isfinite(m_part), "windowed IC is finite")


def test_tune_fold():
    print("\n── single fold tuning ──")
    close, volume, turnover = make_data()
    folds = walk_forward_splits(close.index, n_folds=3, initial_train=0.5)
    grid = make_param_grid((5, 10), (20, 60))

    row = tune_fold(close, folds[0], grid, volume=volume,
                    turnover=turnover, eval_period=5, criterion="ic")
    check(isinstance(row, TuneRow), "returns a TuneRow")
    check(row.params["ic_period"] in (5, 10), "chosen period from the grid")
    check(row.params["ic_window"] in (20, 60), "chosen window from the grid")
    check(np.isfinite(row.train_ic) and np.isfinite(row.test_ic),
          "train and test IC are finite")
    check(row.train_start == folds[0].train_index[0], "train start recorded")
    check(row.test_end == folds[0].test_index[-1], "test end recorded")

    # 测试段 IC 应该和单独重算的一致
    score = ic_weighted_score(close, volume, turnover,
                              ic_period=row.params["ic_period"],
                              ic_window=row.params["ic_window"])
    mean_ic, _, _ = score_in_window(score, close, folds[0].test_index, period=5)
    check(abs(mean_ic - row.test_ic) < 1e-12, "test IC matches an independent run")

    # 换一个挑选标准仍然能跑
    row_icir = tune_fold(close, folds[0], grid, volume=volume,
                         turnover=turnover, criterion="icir")
    check(np.isfinite(row_icir.train_icir), "icir criterion works")

    # 空网格/非法标准
    for kwargs in [dict(grid=[]), dict(grid=grid, criterion="sharpe")]:
        try:
            tune_fold(close, folds[0], kwargs.pop("grid"), **kwargs,
                      volume=volume, turnover=turnover)
            ok = False
        except ValueError:
            ok = True
        check(ok, f"rejects invalid tune_fold {kwargs}")


def test_no_lookahead():
    print("\n── no look-ahead ──")
    close, volume, turnover = make_data(n_days=260)
    cutoff = close.index[200]
    truncated = close.loc[:cutoff]

    full_score = ic_weighted_score(close, volume, turnover,
                                   ic_period=10, ic_window=30)
    cut_score = ic_weighted_score(truncated, volume.loc[:cutoff],
                                  turnover.loc[:cutoff],
                                  ic_period=10, ic_window=30)

    # 前 cutoff - ic_period 天用不到 cutoff 之后的任何价格
    safe = close.index[:195]
    a = full_score.reindex(safe)
    b = cut_score.reindex(safe)
    check(np.allclose(a.fillna(0.0).to_numpy(), b.fillna(0.0).to_numpy(),
                      atol=1e-12),
          "truncating the tail leaves earlier scores unchanged")


def test_walk_forward_and_summary():
    print("\n── walk-forward + summary ──")
    close, volume, turnover = make_data()
    grid = make_param_grid((5, 20), (20, 60))
    rows = walk_forward_tune(close, grid, n_folds=3, initial_train=0.5,
                             volume=volume, turnover=turnover,
                             eval_period=5, criterion="ic")
    check(len(rows) == 3, "one row per fold")
    check([r.fold for r in rows] == [0, 1, 2], "fold ids in order")
    check(all(np.isfinite(r.test_ic) for r in rows), "all test IC finite")

    summary = summarize_tuning(rows)
    check(summary["n_folds"] == 3, "summary fold count")
    check(-1.0 <= summary["mean_oos_ic"] <= 1.0, "mean OOS IC in range")
    check(0.0 <= summary["win_rate"] <= 1.0, "win rate in [0, 1]")
    check(sum(summary["param_counts"].values()) == 3, "param counts sum to folds")
    check(0 <= summary["positive_folds"] <= 3, "positive folds within range")

    report = format_tuning_report(rows, summary)
    for token in ["walk-forward", "样本外平均 IC", "固定参数平均 IC", "选中"]:
        check(token in report, f"report contains {token!r}")

    # 空的 summary 要报错
    try:
        summarize_tuning([])
        ok = False
    except ValueError:
        ok = True
    check(ok, "empty tuning result rejected")

    # 人为构造两折：一折调参更好，一折更差 → 胜率 50%
    mk = lambda f, oos, base: TuneRow(
        fold=f, train_start=close.index[0], train_end=close.index[10],
        test_start=close.index[20], test_end=close.index[30],
        params={"ic_period": 5, "ic_window": 20},
        train_ic=0.05, train_icir=0.2, test_ic=oos,
        test_icir=0.1, baseline_test_ic=base, baseline_test_icir=0.1,
    )
    fake = [mk(0, 0.08, 0.05), mk(1, 0.01, 0.04)]
    s2 = summarize_tuning(fake)
    check(abs(s2["win_rate"] - 0.5) < 1e-12, "win rate counts fold-by-fold wins")
    check(abs(s2["mean_uplift"] - 0.0) < 1e-12, "mean uplift = mean(oos - base)")
    check(s2["distinct_params"] == 1, "same params across folds counted once")


def main():
    print("=" * 62)
    print("  IC tuning tests")
    print("=" * 62)

    test_grid()
    test_splits()
    test_ic_window()
    test_tune_fold()
    test_no_lookahead()
    test_walk_forward_and_summary()

    print(f"\n  Result: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
