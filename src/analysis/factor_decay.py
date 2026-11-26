"""
因子衰减与换手率 —— 持有期到底选多长。

前面的 IC 分析都是固定一个持有期看均值。真正落到策略上还有一个
问题：因子能预测的是"未来几天的收益"，不是"未来二十天"。持有期
拉长有两个相反的效应：

    收益侧   因子衰减 —— 持有越久, 信号里那点信息被市场消化得越多
    成本侧   换手下降 —— 调仓频率变低, 每年交的佣金/印花税/滑点更少

这两条曲线交叉的地方才是合适的持有期。本模块给出量化这两侧的工具：

    decay_curve()          不同 horizon 的 IC
    top_set_turnover()     按该 horizon 调仓时的单边换手率
    holding_period_table() 把 IC、多空收益、换手、成本放在一张表里
    ic_half_life()         IC 衰减到一半需要几个交易日

约定和 factor_research 一致：宽表, index=交易日, columns=股票代码,
因子值越大越看好。

一个需要提醒的地方：h > 1 的未来收益是重叠的，IC 序列因此有
自相关，这里只给点估计，显著性判断用 ic_inference 里那套。
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .factor_research import (
    compute_forward_returns,
    compute_ic,
    quantile_returns,
)

__all__ = [
    "DecayRow",
    "decay_curve",
    "top_set_turnover",
    "holding_period_table",
    "ic_half_life",
    "format_decay_report",
]


# ===================================================================
# IC 衰减曲线
# ===================================================================

def decay_curve(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizons: Sequence[int] = (1, 2, 3, 5, 10, 20),
    method: str = "spearman",
    min_obs: int = 5,
) -> pd.DataFrame:
    """
    对每个 horizon 算一遍横截面 IC。

    Returns
    -------
    DataFrame, index=horizon, columns:
        mean_ic / std_ic / icir / t_stat / n_samples
    """
    rows = {}
    for h in horizons:
        if h <= 0:
            raise ValueError("horizons must be positive")
        fwd = compute_forward_returns(close, periods=h)
        res = compute_ic(factor, fwd, method=method, min_obs=min_obs)
        rows[int(h)] = {
            "mean_ic": res.mean_ic,
            "std_ic": res.std_ic,
            "icir": res.icir,
            "t_stat": res.t_stat,
            "n_samples": res.n_samples,
        }

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "horizon"
    return out.sort_index()


def ic_half_life(curve: pd.DataFrame) -> Optional[float]:
    """
    IC 衰减到首期一半所需的交易日数（线性插值）。

    曲线是离散的，这里只做折线近似；过期后如果 IC 已经反号或
    一直没降到一半，返回 None。h=1 的 IC 为负时不认为有可用的
    衰减半衰期。
    """
    if curve.empty or "mean_ic" not in curve.columns:
        return None

    hs = np.asarray(curve.index, dtype=float)
    ics = np.asarray(curve["mean_ic"], dtype=float)
    if len(hs) < 2 or not np.isfinite(ics[0]) or ics[0] <= 0:
        return None

    target = ics[0] / 2.0
    for i in range(1, len(hs)):
        if not np.isfinite(ics[i]):
            continue
        if ics[i] <= target:
            # 在 (h[i-1], ic[i-1]) 和 (h[i], ic[i]) 之间插值
            num = ics[i - 1] - target
            den = ics[i - 1] - ics[i]
            if den <= 0:
                return float(hs[i])
            return float(hs[i - 1] + (hs[i] - hs[i - 1]) * num / den)
    return None


# ===================================================================
# 换手率
# ===================================================================

def _top_set(f: pd.Series, frac: float, largest: bool) -> set:
    """取因子最高（或最低）的一批股票，数量按比例向上取整"""
    f = f.dropna()
    n = int(len(f) * frac)
    if n < 1:
        n = 1
    return set(f.nlargest(n).index if largest else f.nsmallest(n).index)


def top_set_turnover(
    factor: pd.DataFrame,
    holding_days: int,
    top_frac: float = 0.2,
    include_bottom: bool = True,
) -> float:
    """
    按 holding_days 调仓时，多头组合（以及可选的空头组合）的单边换手率。

    单边换手率的定义：

        turnover_t = 1 - |S_t ∩ S_{t-1}| / |S_t|

    0 表示持仓完全没变，1 表示整组换了一遍。多空组合两边都要换，
    这里把多头和空头的换手率取平均；只要多头就把 include_bottom
    设成 False。

    调仓日按 factor.index 每 holding_days 取一个，和"持有 h 天的
    因子"是对应的：h=5 的因子每 5 个交易日才需要重新选一次股。
    """
    if holding_days <= 0:
        raise ValueError("holding_days must be positive")
    if not 0 < top_frac < 1:
        raise ValueError("top_frac must be in (0, 1)")

    dates = factor.index[::holding_days]
    prev_top = None
    prev_bottom = None
    values = []

    for dt in dates:
        f = factor.loc[dt]
        if f.notna().sum() < 4:
            continue

        cur_top = _top_set(f, top_frac, largest=True)
        cur_bottom = _top_set(f, top_frac, largest=False)

        if prev_top is not None and cur_top:
            to = 1.0 - len(cur_top & prev_top) / len(cur_top)
            if include_bottom and prev_bottom is not None and cur_bottom:
                to_b = 1.0 - len(cur_bottom & prev_bottom) / len(cur_bottom)
                to = 0.5 * (to + to_b)
            values.append(to)

        prev_top = cur_top
        prev_bottom = cur_bottom

    if not values:
        return 0.0
    return float(np.mean(values))


# ===================================================================
# 持有期对照表
# ===================================================================

class DecayRow(dict):
    """一个 horizon 的汇总结果（保持 dict 方便直接转 DataFrame）"""


def holding_period_table(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizons: Sequence[int] = (1, 2, 3, 5, 10, 20),
    top_frac: float = 0.2,
    cost_bps: float = 15.0,
    n_quantiles: int = 5,
    method: str = "spearman",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    把"因子衰减"和"换手成本"放在同一张表里。

    每一行的含义：

        mean_ic      该持有期的 IC 均值
        long_short   每期多空收益（Q_high - Q_low）的均值
        gross_ann    多空收益按 252/h 简单年化
        turnover     按该持有期调仓的单边换手率
        cost_ann     年化交易成本 = 2 × 换手 × cost_bps × (252/h)
        net_ann      gross_ann - cost_ann

    成本按两倍换手算：一次调仓要卖出一边、买进另一边，两边都交。
    单边成本 cost_bps 默认 15bp，大致覆盖佣金、印花税和一点滑点，
    实际值应该按自己的账户和标的改。

    注意 gross_ann 只是把每期多空收益线性年化，没有复利、没有
    考虑做空的融券成本，是"因子本身的毛收益"，不是可直接交易的
    策略净值。
    """
    if factor.shape[1] < n_quantiles:
        raise ValueError("not enough stocks for n_quantiles")

    rows = {}
    for h in horizons:
        if h <= 0:
            raise ValueError("horizons must be positive")

        fwd = compute_forward_returns(close, periods=h)
        ic = compute_ic(factor, fwd, method=method)
        q = quantile_returns(factor, fwd, n_quantiles=n_quantiles)

        if q.empty:
            ls = np.nan
        else:
            high = q.columns[-1]
            low = q.columns[0]
            ls = float((q[high] - q[low]).mean())

        gross = ls * periods_per_year / h if np.isfinite(ls) else np.nan
        to = top_set_turnover(factor, holding_days=h, top_frac=top_frac)
        cost = 2.0 * to * (cost_bps / 10000.0) * periods_per_year / h
        net = gross - cost if np.isfinite(gross) else np.nan

        rows[int(h)] = {
            "mean_ic": ic.mean_ic,
            "long_short": ls,
            "gross_ann": gross,
            "turnover": to,
            "cost_ann": cost,
            "net_ann": net,
        }

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "horizon"
    return out.sort_index()


# ===================================================================
# 文本报告
# ===================================================================

def format_decay_report(
    tables: Dict[str, pd.DataFrame],
    cost_bps: float = 15.0,
    top_frac: float = 0.2,
) -> str:
    """把多个因子的持有期对照表拼成一段文本"""
    lines = []
    lines.append("=" * 78)
    lines.append("  因子衰减与持有期")
    lines.append("=" * 78)
    lines.append(f"  多空分位: 最高/最低 {top_frac:.0%}  "
                 f"单边成本: {cost_bps:.1f} bp  年化按 252 日")

    for name, df in tables.items():
        lines.append("")
        lines.append(f"  {name}")
        lines.append("  " + "-" * 74)
        lines.append(
            f"  {'持有期':>6}{'mean IC':>10}{'多空/期':>10}"
            f"{'毛年化':>10}{'换手':>8}{'成本':>10}{'净年化':>10}")
        for h, row in df.iterrows():
            lines.append(
                f"  {int(h):>6}{row['mean_ic']:>10.4f}"
                f"{row['long_short']:>10.2%}{row['gross_ann']:>10.1%}"
                f"{row['turnover']:>8.1%}{row['cost_ann']:>10.1%}"
                f"{row['net_ann']:>10.1%}")

    lines.append("")
    lines.append("  说明: 毛年化是多空收益的线性年化, 不含融券成本;")
    lines.append("        净年化已扣 2×换手×单边成本。")
    return "\n".join(lines)
