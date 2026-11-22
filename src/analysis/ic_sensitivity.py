"""
IC 期限敏感性分析。

因子研究里最常见的两个期限:

    信号期限 (lookback)  —— 因子用过去多少天的数据算出来
    评估期限 (horizon)   —— IC 是对未来多少天的收益算的

很多结论对这两个数字非常敏感。一个 20 日动量在 5 日收益上 IC 只有
0.01，换成 20 日收益可能变成 0.05 —— 但这不一定是因子变强了，也可能
只是重叠窗口把样本相关性拉高了。本模块做三件事:

    1. 固定信号期限，扫评估期限 —— 看 IC 随 horizon 怎么变
    2. 固定评估期限，扫信号期限 —— 看因子回忆多长的历史合适
    3. 两者交叉成矩阵，并给出 Newey-West 修正后的 t 值

为什么需要 Newey-West:

    fwd_ret_t = close_{t+H}/close_t - 1

相邻 H 天的样本共享 H-1 天的价格，所以 IC 序列天然有正自相关。
普通 t = mean / (std/sqrt(n)) 把重叠样本当成独立样本，会系统性高估
显著性。Newey-West 用 lag=H-1 的 Bartlett 核估计长期方差，修正后
的 t 值才是能拿去下结论的那个。

所有函数沿用 factor_research 的宽表约定:
    index = 交易日, columns = 股票代码
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analysis.factor_research import (
    compute_forward_returns,
    compute_ic,
    make_factors,
)

__all__ = [
    "ICStats",
    "newey_west_tstat",
    "make_lookback_factor",
    "ic_stats",
    "analyze_horizons",
    "analyze_lookbacks",
    "analyze_grid",
    "stats_table",
    "subperiod_ic",
    "sign_consistency",
    "format_sensitivity_report",
]


# ===================================================================
# 统计量
# ===================================================================

@dataclass
class ICStats:
    """一段 IC 序列的统计量（普通 t 和 Newey-West t 并存）"""

    mean_ic: float
    std_ic: float
    icir: float
    t_stat: float          # 普通 t：假设样本独立
    nw_t_stat: float       # Newey-West 修正 t：容忍 lag 阶自相关
    positive_ratio: float
    n_samples: int

    def __repr__(self) -> str:
        return (f"ICStats(mean={self.mean_ic:.4f}, ICIR={self.icir:.3f}, "
                f"t={self.t_stat:.2f}, nw_t={self.nw_t_stat:.2f}, "
                f"n={self.n_samples})")


def newey_west_tstat(values: Sequence[float], lags: int) -> float:
    """
    均值的 Newey-West t 值。

    长期方差估计:

        v = γ0 + 2 * Σ_{l=1..L} (1 - l/(L+1)) * γl

    其中 γl 是滞后 l 的自协方差（除以 n，不是 n-l）。L=0 时退化
    成普通 t。重叠收益的 lag 取 H-1 就能覆盖重叠结构。

    返回 0.0 表示样本不足或方差非正，调用方按"算不出显著性"处理。
    """
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]

    n = x.size
    if n < 2:
        return 0.0
    if lags < 0:
        raise ValueError("lags must be >= 0")
    lags = min(int(lags), n - 1)

    mean = float(x.mean())
    e = x - mean
    gamma0 = float(e @ e) / n
    var = gamma0

    for lag in range(1, lags + 1):
        gamma = float(e[lag:] @ e[:-lag]) / n
        weight = 1.0 - lag / (lags + 1.0)
        var += 2.0 * weight * gamma

    if var <= 0.0:
        return 0.0
    return mean / np.sqrt(var / n)


def ic_stats(
    factor: pd.DataFrame,
    forward_ret: pd.DataFrame,
    *,
    nw_lags: Optional[int] = None,
    method: str = "spearman",
    min_obs: int = 5,
) -> ICStats:
    """
    计算 IC 序列并汇总。

    nw_lags 不传时取 0（等价于普通 t）。做期限扫描时调用方会显式
    传 horizon-1，因为评估期越长重叠越严重。
    """
    result = compute_ic(factor, forward_ret, method=method, min_obs=min_obs)
    lags = 0 if nw_lags is None else int(nw_lags)
    nw_t = newey_west_tstat(result.ic_series, lags) if result.n_samples else 0.0

    return ICStats(
        mean_ic=result.mean_ic,
        std_ic=result.std_ic,
        icir=result.icir,
        t_stat=result.t_stat,
        nw_t_stat=nw_t,
        positive_ratio=result.positive_ratio,
        n_samples=result.n_samples,
    )


# ===================================================================
# 因子构造
# ===================================================================

def make_lookback_factor(
    close: pd.DataFrame,
    lookback: int,
    kind: str = "momentum",
) -> pd.DataFrame:
    """
    按指定信号期限构造因子。

        kind="momentum"  →  close / close.shift(L) - 1
        kind="reversal"  →  取负号（A 股短期反转的方向）

    L=0 无意义，直接报错。
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if kind not in ("momentum", "reversal"):
        raise ValueError("kind must be 'momentum' or 'reversal'")

    ret = close.pct_change(lookback)
    return ret if kind == "momentum" else -ret


# ===================================================================
# 两个方向的扫描
# ===================================================================

def analyze_horizons(
    close: pd.DataFrame,
    *,
    factors: Optional[Iterable[str]] = None,
    horizons: Sequence[int] = (1, 3, 5, 10, 20),
    volume: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.DataFrame] = None,
    method: str = "spearman",
    min_obs: int = 5,
) -> Dict[str, Dict[int, ICStats]]:
    """
    固定因子定义，扫评估期限。

    返回 {factor_name: {horizon: ICStats}}。每个 horizon 的 t 值都
    用该 horizon 对应的重叠阶数修正。
    """
    library = make_factors(close, volume=volume, turnover=turnover)
    names = list(factors) if factors else [n for n in
                                           ("momentum_20", "reversal_5", "vol_20")
                                           if n in library]

    unknown = [n for n in names if n not in library]
    if unknown:
        raise ValueError(f"unknown factors: {', '.join(unknown)}")

    out: Dict[str, Dict[int, ICStats]] = {}
    for name in names:
        out[name] = {}
        for h in _check_periods(horizons, "horizons"):
            fwd = compute_forward_returns(close, periods=h)
            out[name][h] = ic_stats(
                library[name], fwd,
                nw_lags=h - 1, method=method, min_obs=min_obs,
            )
    return out


def analyze_lookbacks(
    close: pd.DataFrame,
    *,
    lookbacks: Sequence[int] = (5, 10, 20, 40, 60),
    horizon: int = 5,
    kind: str = "momentum",
    method: str = "spearman",
    min_obs: int = 5,
) -> Dict[int, ICStats]:
    """固定评估期限，扫信号期限。"""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    fwd = compute_forward_returns(close, periods=horizon)
    out: Dict[int, ICStats] = {}
    for lookback in _check_periods(lookbacks, "lookbacks"):
        factor = make_lookback_factor(close, lookback, kind=kind)
        out[lookback] = ic_stats(
            factor, fwd,
            nw_lags=horizon - 1, method=method, min_obs=min_obs,
        )
    return out


def analyze_grid(
    close: pd.DataFrame,
    *,
    lookbacks: Sequence[int] = (5, 10, 20, 40),
    horizons: Sequence[int] = (1, 3, 5, 10, 20),
    kind: str = "momentum",
    method: str = "spearman",
    min_obs: int = 5,
) -> Dict[Tuple[int, int], ICStats]:
    """
    信号期限 × 评估期限的完整矩阵。

    返回 {(lookback, horizon): ICStats}。同一个 lookback 的因子只算
    一次，复用到所有 horizon —— 网格大时这能省掉一半以上的计算。
    """
    lbs = _check_periods(lookbacks, "lookbacks")
    hs = _check_periods(horizons, "horizons")

    fwd_by_h = {h: compute_forward_returns(close, periods=h) for h in hs}
    out: Dict[Tuple[int, int], ICStats] = {}

    for lb in lbs:
        factor = make_lookback_factor(close, lb, kind=kind)
        for h in hs:
            out[(lb, h)] = ic_stats(
                factor, fwd_by_h[h],
                nw_lags=h - 1, method=method, min_obs=min_obs,
            )
    return out


# ===================================================================
# 表格与稳定性
# ===================================================================

def stats_table(
    stats: Dict,
    rows: Sequence,
    cols: Sequence,
    field: str = "mean_ic",
) -> pd.DataFrame:
    """
    把嵌套的统计量字典摊成宽表。

    stats 支持两种形状:
        {row_key: {col_key: ICStats}}
        {(row_key, col_key): ICStats}

    缺失的组合填 NaN，方便直接 .to_string() 打印。
    """
    data = []
    for r in rows:
        line = []
        for c in cols:
            cell = stats.get((r, c)) if (r, c) in stats else stats.get(r, {}).get(c)
            line.append(getattr(cell, field) if cell is not None else np.nan)
        data.append(line)
    return pd.DataFrame(data, index=list(rows), columns=list(cols), dtype=float)


def subperiod_ic(
    factor: pd.DataFrame,
    forward_ret: pd.DataFrame,
    n_segments: int = 4,
    *,
    method: str = "spearman",
    min_obs: int = 5,
) -> list:
    """
    把样本按时间等分成 n 段，分别算平均 IC。

    全样本 IC 是 0.03，但四段分别是 +0.06/+0.05/-0.01/-0.02 的话，
    这个因子近半段已经在失效 —— 只看全样本是看不出来的。
    """
    if n_segments < 2:
        raise ValueError("n_segments must be >= 2")

    result = compute_ic(factor, forward_ret, method=method, min_obs=min_obs)
    ic = result.ic_series
    if ic.empty:
        return []

    # 按位置切，不按日历切 —— 停牌多的区间也不会出现空段
    chunks = np.array_split(np.arange(len(ic)), n_segments)
    return [float(ic.iloc[idx].mean()) if len(idx) else np.nan
            for idx in chunks]


def sign_consistency(segment_means: Sequence[float], full_mean: float) -> float:
    """
    分段均值和全样本均值同号的占比。

    0.75 表示 4 段里有 3 段方向一致；0.5 基本可以当成不稳定。
    """
    vals = [v for v in segment_means if np.isfinite(v)]
    if not vals or not np.isfinite(full_mean) or full_mean == 0:
        return 0.0
    same = sum(1 for v in vals if (v > 0) == (full_mean > 0))
    return same / len(vals)


# ===================================================================
# 文本报告
# ===================================================================

def format_sensitivity_report(
    horizon_stats: Dict[str, Dict[int, ICStats]],
    grid_stats: Dict[Tuple[int, int], ICStats],
    lookbacks: Sequence[int],
    horizons: Sequence[int],
    stability: Optional[Dict[str, Dict]] = None,
    kind: str = "momentum",
) -> str:
    """把三张表拼成一段可读文本。"""
    lines = []
    lines.append("=" * 66)
    lines.append("  IC 期限敏感性")
    lines.append("=" * 66)

    # ---- 表 1: 因子 × 评估期限 (平均 IC / NW t) ----
    names = list(horizon_stats.keys())
    hs = sorted({h for d in horizon_stats.values() for h in d})

    lines.append("")
    lines.append("  1) 因子 × 评估期限：平均 RankIC")
    lines.append("     " + " " * 16 + "".join(f"{h:>8}d" for h in hs))
    for name in names:
        vals = [horizon_stats[name].get(h) for h in hs]
        row = "".join(f"{v.mean_ic:>9.4f}" if v else f"{'--':>9}"
                      for v in vals)
        lines.append(f"     {name:<16}{row}")

    lines.append("")
    lines.append("  Newey-West t（lag = horizon - 1；括号内是未修正 t）")
    lines.append("     " + " " * 16 + "".join(f"{h:>15}d" for h in hs))
    for name in names:
        vals = [horizon_stats[name].get(h) for h in hs]
        row = "".join(
            f"{v.nw_t_stat:>9.2f}({v.t_stat:>5.2f})" if v else f"{'--':>15}"
            for v in vals
        )
        lines.append(f"     {name:<16}{row}")

    # ---- 表 2: 信号期限 × 评估期限 ----
    lines.append("")
    lines.append(f"  2) {kind} 因子：信号期限 × 评估期限，平均 RankIC")
    lines.append("     " + " " * 8 + "".join(f"{h:>8}d" for h in horizons))
    table = stats_table(grid_stats, lookbacks, horizons, "mean_ic")
    for lb in lookbacks:
        row = "".join(f"{table.loc[lb, h]:>9.4f}" for h in horizons)
        lines.append(f"     {lb:>4}d   {row}")

    # ---- 表 3: 分段稳定性 ----
    if stability:
        lines.append("")
        lines.append("  3) 分段稳定性（全样本 vs 各段平均 IC）")
        for name, info in stability.items():
            segs = info["segments"]
            seg_txt = " ".join(f"{v:+.4f}" if np.isfinite(v) else "  --  "
                               for v in segs)
            lines.append(
                f"     {name:<16} full={info['full']:+.4f}  "
                f"same-sign={info['consistency']:.0%}  [{seg_txt}]"
            )

    lines.append("")
    lines.append("  读法:")
    lines.append("  - IC 随 horizon 变大很可能是重叠样本造成的假象，先看 NW t。")
    lines.append("  - 信号期限扫过去最优值跳来跳去，说明这个池子上没有稳定窗口。")
    lines.append("  - 分段方向不一致的因子，全样本 IC 再高也不适合直接上策略。")
    lines.append("=" * 66)
    return "\n".join(lines)


def _check_periods(values: Sequence[int], name: str) -> list:
    out = []
    for v in values:
        iv = int(v)
        if iv <= 0:
            raise ValueError(f"{name} must be positive")
        if iv not in out:
            out.append(iv)
    if not out:
        raise ValueError(f"{name} is empty")
    return out
