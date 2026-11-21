"""
IC 加权窗口的样本外调参 (Phase 20)。

滚动 IC 加权有两个超参数：
    ic_period  —— 用多少天的前向收益去估计 IC
    ic_window  —— 拿过去多少个 IC 求平均

在整段历史上网格搜索一组"最好"的参数，再把同一组参数拿去汇报收益，
是回测里最典型的过拟合：参数是看着答案选的。

这里的做法是 walk-forward：

    |---- 训练段 ----|-- 测试段 --|
    |-------- 训练段 --------|-- 测试段 --|
    |------------ 训练段 ------------|-- 测试段 --|

每个折只在训练段上挑参数（按训练段 IC/ICIR），然后把这组参数原样
用到紧挨着的测试段，只记录测试段的表现。最后和一组固定参数
（默认 ic_period=20, ic_window=60）逐折对比。

注意两件事：

1. 参数选择只看训练段的 IC，测试段完全不参与。
2. 打分本身是严格因果的：ic_weighted_score 在 t 日只用 t-ic_period
   以前已知的 IC，测试段内部也不偷看未来。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..analysis.factor_research import compute_forward_returns, compute_ic
from .factor_selection import DEFAULT_FACTORS, ic_weighted_score

__all__ = [
    "WalkForwardFold",
    "TuneRow",
    "make_param_grid",
    "walk_forward_splits",
    "score_in_window",
    "tune_fold",
    "walk_forward_tune",
    "summarize_tuning",
    "format_tuning_report",
]

CRITERIA = ("ic", "icir", "t")


@dataclass
class WalkForwardFold:
    """一个折的训练/测试区间（都是 DatetimeIndex 的切片）"""

    fold: int
    train_index: pd.Index
    test_index: pd.Index

    @property
    def train_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return self.train_index[0], self.train_index[-1]

    @property
    def test_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return self.test_index[0], self.test_index[-1]


@dataclass
class TuneRow:
    """一个折的调参结果"""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: Dict[str, int]
    train_ic: float
    train_icir: float
    test_ic: float
    test_icir: float
    baseline_test_ic: float = float("nan")
    baseline_test_icir: float = float("nan")


def make_param_grid(
    ic_periods: Sequence[int] = (5, 10, 20, 40),
    ic_windows: Sequence[int] = (20, 40, 60, 120),
) -> List[Dict[str, int]]:
    """笛卡尔积。参数重复或非正数时报错，免得静默跑出重复行。"""
    periods = [int(p) for p in ic_periods]
    windows = [int(w) for w in ic_windows]
    if not periods or not windows:
        raise ValueError("参数网格不能为空")
    if any(p <= 0 for p in periods) or any(w <= 0 for w in windows):
        raise ValueError("ic_period / ic_window 必须为正")
    if len(set(periods)) != len(periods) or len(set(windows)) != len(windows):
        raise ValueError("参数网格里有重复值")

    return [
        {"ic_period": p, "ic_window": w}
        for p, w in product(periods, windows)
    ]


def walk_forward_splits(
    index: pd.Index,
    n_folds: int = 3,
    initial_train: float = 0.5,
    mode: str = "expanding",
    min_train: int = 60,
    min_test: int = 20,
) -> List[WalkForwardFold]:
    """把时间轴切成"训练 → 测试"若干折。

    Parameters
    ----------
    index          : 交易日索引（升序）
    n_folds        : 测试段个数
    initial_train  : 第一个训练段占全样本的比例 (0, 1)
    mode           : "expanding" 训练段逐折变长；
                     "rolling"   训练段长度固定，整体向后滚
    min_train      : 训练段最少多少天，不够直接报错
    min_test       : 每个测试段最少多少天
    """
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if mode not in ("expanding", "rolling"):
        raise ValueError("mode must be 'expanding' or 'rolling'")
    if not 0.0 < initial_train < 1.0:
        raise ValueError("initial_train must be in (0, 1)")

    n = len(index)
    split = int(n * initial_train)
    if split < min_train:
        raise ValueError(f"训练段只有 {split} 天，至少需要 {min_train} 天")
    if n - split < n_folds * min_test:
        raise ValueError(
            f"测试段只有 {n - split} 天，{n_folds} 折每折至少 {min_test} 天"
        )

    # 测试段尽量等长；余数摊到前面几折，保证不重叠、不遗漏
    total_test = n - split
    base = total_test // n_folds
    extra = total_test % n_folds

    folds: List[WalkForwardFold] = []
    start = split
    for k in range(n_folds):
        size = base + (1 if k < extra else 0)
        stop = start + size if k < n_folds - 1 else n

        train = index[:start] if mode == "expanding" else index[start - split:start]
        if len(train) < min_train:
            raise ValueError(f"第 {k} 折训练段太短: {len(train)}")
        if stop - start < min_test:
            raise ValueError(f"第 {k} 折测试段太短: {stop - start}")

        folds.append(WalkForwardFold(k, train, index[start:stop]))
        start = stop

    return folds


def score_in_window(score: pd.DataFrame, close: pd.DataFrame, index,
                    period: int) -> Tuple[float, float, int]:
    """在给定日期区间上算合成分数的 RankIC，返回 (mean_ic, icir, n)。"""
    fwd = compute_forward_returns(close, periods=period)
    sub = score.reindex(index)
    ic = compute_ic(sub, fwd.reindex(index))
    return ic.mean_ic, ic.icir, ic.n_samples


def _criterion_value(ic_result, criterion: str) -> float:
    if criterion == "ic":
        return ic_result.mean_ic
    if criterion == "icir":
        return ic_result.icir
    if criterion == "t":
        return ic_result.t_stat
    raise ValueError(f"未知的挑选标准: {criterion}")


def tune_fold(
    close: pd.DataFrame,
    fold: WalkForwardFold,
    grid: Sequence[Dict[str, int]],
    *,
    volume: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.DataFrame] = None,
    factors: Optional[Iterable[str]] = None,
    eval_period: int = 5,
    criterion: str = "ic",
    winsorize: bool = False,
    industry=None,
    log_mktcap: Optional[pd.DataFrame] = None,
    orthogonalize: bool = False,
    baseline: Optional[Dict[str, int]] = None,
) -> TuneRow:
    """在一个折上跑完整网格，用训练段挑参数，只把测试段 IC 记进结果。"""
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}")
    if not grid:
        raise ValueError("参数网格不能为空")

    names = list(factors) if factors else list(DEFAULT_FACTORS)
    fwd = compute_forward_returns(close, periods=eval_period)

    best = None          # (criterion_value, params, train_ic, train_icir, score)
    for params in grid:
        score = ic_weighted_score(
            close, volume, turnover,
            factors=names,
            ic_period=int(params["ic_period"]),
            ic_window=int(params["ic_window"]),
            winsorize=winsorize,
            industry=industry,
            log_mktcap=log_mktcap,
            orthogonalize=orthogonalize,
        )

        train_ic = compute_ic(score.reindex(fold.train_index),
                              fwd.reindex(fold.train_index))
        if train_ic.n_samples == 0:
            continue

        val = _criterion_value(train_ic, criterion)
        if not np.isfinite(val):
            continue

        # 稳定排序：同分时保留网格里的先后顺序，结果可复现
        if best is None or val > best[0]:
            best = (val, dict(params), train_ic.mean_ic, train_ic.icir, score)

    if best is None:
        raise ValueError(f"第 {fold.fold} 折没有任何参数能算出训练段 IC")

    _, params, train_ic_val, train_icir, score = best
    test_ic = compute_ic(score.reindex(fold.test_index),
                         fwd.reindex(fold.test_index))

    base_params = dict(baseline or {"ic_period": 20, "ic_window": 60})
    base_score = ic_weighted_score(
        close, volume, turnover,
        factors=names,
        ic_period=int(base_params["ic_period"]),
        ic_window=int(base_params["ic_window"]),
        winsorize=winsorize,
        industry=industry,
        log_mktcap=log_mktcap,
        orthogonalize=orthogonalize,
    )
    base_test = compute_ic(base_score.reindex(fold.test_index),
                           fwd.reindex(fold.test_index))

    return TuneRow(
        fold=fold.fold,
        train_start=fold.train_range[0],
        train_end=fold.train_range[1],
        test_start=fold.test_range[0],
        test_end=fold.test_range[1],
        params=params,
        train_ic=train_ic_val,
        train_icir=train_icir,
        test_ic=test_ic.mean_ic,
        test_icir=test_ic.icir,
        baseline_test_ic=base_test.mean_ic,
        baseline_test_icir=base_test.icir,
    )


def walk_forward_tune(
    close: pd.DataFrame,
    grid: Sequence[Dict[str, int]],
    *,
    n_folds: int = 3,
    initial_train: float = 0.5,
    mode: str = "expanding",
    volume: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.DataFrame] = None,
    factors: Optional[Iterable[str]] = None,
    eval_period: int = 5,
    criterion: str = "ic",
    winsorize: bool = False,
    industry=None,
    log_mktcap: Optional[pd.DataFrame] = None,
    orthogonalize: bool = False,
    baseline: Optional[Dict[str, int]] = None,
) -> List[TuneRow]:
    """逐折调参。返回每个折一行，顺序和测试段一致。

    和逐折调用 tune_fold 的区别：这里的分数只算一次。同一组参数在
    每一折里都是同一条分数序列，折与折只是切不同的日期区间；不缓存
    的话 16 组参数 × 3 折要重复算 48 次因子和滚动 IC，慢得没必要。
    """
    if not grid:
        raise ValueError("参数网格不能为空")
    if criterion not in CRITERIA:
        raise ValueError(f"criterion must be one of {CRITERIA}")

    folds = walk_forward_splits(
        close.index, n_folds=n_folds,
        initial_train=initial_train, mode=mode,
    )

    names = list(factors) if factors else list(DEFAULT_FACTORS)
    fwd = compute_forward_returns(close, periods=eval_period)

    # 每条参数只算一次分数
    base_params = dict(baseline or {"ic_period": 20, "ic_window": 60})
    score_cache: Dict[Tuple[int, int], pd.DataFrame] = {}

    def get_score(params: Dict[str, int]) -> pd.DataFrame:
        key = (int(params["ic_period"]), int(params["ic_window"]))
        if key not in score_cache:
            score_cache[key] = ic_weighted_score(
                close, volume, turnover,
                factors=names,
                ic_period=key[0],
                ic_window=key[1],
                winsorize=winsorize,
                industry=industry,
                log_mktcap=log_mktcap,
                orthogonalize=orthogonalize,
            )
        return score_cache[key]

    base_score = get_score(base_params)

    rows: List[TuneRow] = []
    for fold in folds:
        best = None
        for params in grid:
            score = get_score(params)
            train_ic = compute_ic(score.reindex(fold.train_index),
                                  fwd.reindex(fold.train_index))
            if train_ic.n_samples == 0:
                continue
            val = _criterion_value(train_ic, criterion)
            if not np.isfinite(val):
                continue
            if best is None or val > best[0]:
                best = (val, dict(params), train_ic.mean_ic,
                        train_ic.icir, score)

        if best is None:
            raise ValueError(f"第 {fold.fold} 折没有任何参数能算出训练段 IC")

        _, params, train_ic_val, train_icir, score = best
        test_ic = compute_ic(score.reindex(fold.test_index),
                             fwd.reindex(fold.test_index))
        base_test = compute_ic(base_score.reindex(fold.test_index),
                               fwd.reindex(fold.test_index))

        rows.append(TuneRow(
            fold=fold.fold,
            train_start=fold.train_range[0],
            train_end=fold.train_range[1],
            test_start=fold.test_range[0],
            test_end=fold.test_range[1],
            params=params,
            train_ic=train_ic_val,
            train_icir=train_icir,
            test_ic=test_ic.mean_ic,
            test_icir=test_ic.icir,
            baseline_test_ic=base_test.mean_ic,
            baseline_test_icir=base_test.icir,
        ))

    return rows


def summarize_tuning(rows: Sequence[TuneRow]) -> dict:
    """把逐折结果压成几个能直接读的数。

    win_rate 是逐折比较（调参后测试 IC > 固定参数测试 IC 的折数占比），
    不是统计意义上的胜率，样本通常只有 3-5 折。
    """
    if not rows:
        raise ValueError("没有折结果可以汇总")

    oos = np.array([r.test_ic for r in rows], dtype=float)
    base = np.array([r.baseline_test_ic for r in rows], dtype=float)
    valid = np.isfinite(oos) & np.isfinite(base)

    param_counts: Dict[str, int] = {}
    for r in rows:
        key = f"{r.params['ic_period']}/{r.params['ic_window']}"
        param_counts[key] = param_counts.get(key, 0) + 1

    return {
        "n_folds": len(rows),
        "mean_oos_ic": float(np.nanmean(oos)),
        "median_oos_ic": float(np.nanmedian(oos)),
        "mean_baseline_ic": float(np.nanmean(base)),
        "mean_uplift": float(np.nanmean(oos - base)),
        "win_rate": float(np.mean(oos[valid] > base[valid])) if valid.any() else 0.0,
        "positive_folds": int(np.sum(oos > 0)),
        "param_counts": param_counts,
        "distinct_params": len(param_counts),
    }


def format_tuning_report(
    rows: Sequence[TuneRow],
    summary: Optional[dict] = None,
    *,
    criterion: str = "ic",
) -> str:
    """终端报告：逐折挑选结果 + 样本外对比 + 参数稳定性。"""
    summary = summary or summarize_tuning(rows)

    lines = []
    lines.append("=" * 72)
    lines.append("  IC 参数 walk-forward 调参 (样本外)")
    lines.append("=" * 72)
    lines.append(f"  选择标准: 训练段 {criterion}")
    lines.append("")
    lines.append(f"  {'折':>2}  {'训练区间':<23} {'测试区间':<23} "
                 f"{'参数':>8} {'训练IC':>8} {'测试IC':>8} {'基线IC':>8}")
    lines.append("  " + "-" * 70)

    for r in rows:
        tr = f"{r.train_start.date()}~{r.train_end.date()}"
        te = f"{r.test_start.date()}~{r.test_end.date()}"
        params = f"{r.params['ic_period']}/{r.params['ic_window']}"
        lines.append(
            f"  {r.fold:>2}  {tr:<23} {te:<23} {params:>8} "
            f"{r.train_ic:>+8.4f} {r.test_ic:>+8.4f} {r.baseline_test_ic:>+8.4f}"
        )

    lines.append("")
    lines.append(f"  样本外平均 IC : {summary['mean_oos_ic']:+.4f}")
    lines.append(f"  固定参数平均 IC: {summary['mean_baseline_ic']:+.4f}")
    lines.append(f"  平均提升       : {summary['mean_uplift']:+.4f}")
    lines.append(f"  逐折胜出       : {summary['win_rate']:.0%} "
                 f"({int(round(summary['win_rate'] * summary['n_folds']))}"
                 f"/{summary['n_folds']})")
    lines.append(f"  正 IC 折数    : {summary['positive_folds']}/{summary['n_folds']}")
    lines.append("")

    counts = summary["param_counts"]
    picks = "  ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
    lines.append(f"  各折选中的参数 (ic_period/ic_window): {picks}")
    lines.append(f"  不同参数个数  : {summary['distinct_params']}"
                 f"/{summary['n_folds']}")
    if summary["distinct_params"] == summary["n_folds"] and summary["n_folds"] > 1:
        lines.append("  提示: 每折都选了不同参数，说明最优窗口本身不稳定，")
        lines.append("        固定参数并不比逐折挑选差多少——别只看提升。")
    lines.append("=" * 72)

    return "\n".join(lines)
