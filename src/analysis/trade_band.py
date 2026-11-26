"""
调仓缓冲带 —— 少换一点，代价是多少。

Phase 25 把换手率算出来之后，很自然的问题是：能不能少换一点？
最直接的办法是排名掉出前 N 就卖、空位再补，这样每次调仓都要动一批
边缘股票。另一种做法是给持仓加一个缓冲带：

    现有持仓只要还在排名前 N × band 名以内，就不卖；
    只有掉出这个范围才换掉，空出的名额再从排名最前面补。

band = 1.0 等价于没有缓冲（前 N 名就是持仓），band = 1.5 表示
"前 1.5N 名内都留着"。band 越大越懒得换，换手下降，但也可能
留着已经变差的股票。

本模块只做离线估算：用因子分层/持仓集合模拟，不接实盘撮合。
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Set

import numpy as np
import pandas as pd

from .factor_research import compute_forward_returns

__all__ = [
    "banded_holdings",
    "banded_portfolio",
    "band_comparison",
    "format_band_report",
]


def _pick_top(ranked: pd.Series, n: int) -> Set[str]:
    return set(ranked.index[:n])


def banded_holdings(
    factor: pd.DataFrame,
    holding_days: int,
    top_frac: float = 0.2,
    band: float = 1.5,
):
    """
    逐期维护持仓集合，返回 (调仓日, 持仓集合, 相对上期的换手率) 列表。

    规则：
        n        = 每期持有只数（按 top_frac 向上取整）
        keep_n   = n * band，向上取整，至少 n
        现有持仓里只要还排在前 keep_n 名就保留；
        不足 n 只时，从排名最靠前、且不在持仓里的股票补足。

    第一期的换手率定义为 1.0（从空仓建仓）。
    """
    if holding_days <= 0:
        raise ValueError("holding_days must be positive")
    if not 0 < top_frac < 1:
        raise ValueError("top_frac must be in (0, 1)")
    if band < 1.0:
        raise ValueError("band must be >= 1.0")

    dates = factor.index[::holding_days]
    holdings: Set[str] = set()
    rows = []

    for dt in dates:
        f = factor.loc[dt].dropna()
        if len(f) < 2:
            continue

        ranked = f.sort_values(ascending=False)
        n = max(1, int(len(ranked) * top_frac))
        keep_n = max(n, int(np.ceil(n * band)))

        if not holdings:
            new = _pick_top(ranked, n)
            turnover = 1.0
        else:
            keep_pool = set(ranked.index[:keep_n])
            kept = holdings & keep_pool
            need = n - len(kept)
            if need > 0:
                fillers = [s for s in ranked.index if s not in kept]
                kept |= set(fillers[:need])
            new = kept
            turnover = 1.0 - len(new & holdings) / len(new)

        rows.append((dt, new, turnover))
        holdings = new

    return rows


def banded_portfolio(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    holding_days: int,
    top_frac: float = 0.2,
    band: float = 1.5,
    cost_bps: float = 15.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    按缓冲带规则模拟多空组合，逐期记录。

    多头持仓取因子排名最前的一批，空头取最后的一批。两边的缓冲带
    规则相同。每期收益 = 持仓股票在接下来 holding_days 个交易日的
    等权平均收益；成本按 2 × 换手 × 单边 bps 扣。

    返回 DataFrame，index = 调仓日，columns:
        long_ret / short_ret / ls_ret / turnover / cost
    """
    rows = []
    idx = []

    long_rows = banded_holdings(factor, holding_days, top_frac, band)
    # 空头：把因子取负，复用同一套状态机
    short_rows = banded_holdings(-factor, holding_days, top_frac, band)

    fwd = compute_forward_returns(close, holding_days)

    short_map = {dt: (hold, to) for dt, hold, to in short_rows}

    for dt, long_hold, long_to in long_rows:
        if dt not in short_map or dt not in fwd.index:
            continue
        short_hold, short_to = short_map[dt]

        r = fwd.loc[dt]
        lcols = [c for c in long_hold if c in r.index and pd.notna(r[c])]
        scols = [c for c in short_hold if c in r.index and pd.notna(r[c])]

        lr = float(r[lcols].mean()) if lcols else np.nan
        sr = float(r[scols].mean()) if scols else np.nan
        if not (np.isfinite(lr) and np.isfinite(sr)):
            continue

        turnover = 0.5 * (long_to + short_to)
        cost = 2.0 * turnover * cost_bps / 10000.0

        rows.append({
            "long_ret": lr,
            "short_ret": sr,
            "ls_ret": lr - sr,
            "turnover": turnover,
            "cost": cost,
        })
        idx.append(dt)

    if not rows:
        return pd.DataFrame(
            columns=["long_ret", "short_ret", "ls_ret", "turnover", "cost"])

    return pd.DataFrame(rows, index=pd.Index(idx))


def band_comparison(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    holding_days: int,
    top_frac: float = 0.2,
    bands: Sequence[float] = (1.0, 1.2, 1.5, 2.0),
    cost_bps: float = 15.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    跑一组缓冲带，输出换手 / 毛收益 / 成本 / 净收益的对照表。

    毛年化 = 每期多空收益均值 × 252 / h
    成本年化 = 每期成本均值 × 252 / h
    净年化 = 毛年化 - 成本年化
    """
    rows = {}

    for band in bands:
        df = banded_portfolio(
            factor, close, holding_days,
            top_frac=top_frac, band=band, cost_bps=cost_bps,
        )
        if df.empty:
            continue

        per_year = periods_per_year / holding_days
        gross = float(df["ls_ret"].mean()) * per_year
        cost = float(df["cost"].mean()) * per_year

        rows[band] = {
            "turnover": float(df["turnover"].mean()),
            "gross_ann": gross,
            "cost_ann": cost,
            "net_ann": gross - cost,
            "periods": len(df),
        }

    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "band"
    return out.sort_index()


def format_band_report(
    tables: Dict[str, pd.DataFrame],
    holding_days: int,
    cost_bps: float = 15.0,
) -> str:
    """把多个因子的缓冲带对照表拼成文本"""
    lines = []
    lines.append("=" * 72)
    lines.append("  调仓缓冲带 (no-trade band)")
    lines.append("=" * 72)
    lines.append(f"  持有期 {holding_days} 日  单边成本 {cost_bps:.1f} bp  "
                 f"band=1.0 表示无缓冲")

    for name, df in tables.items():
        lines.append("")
        lines.append(f"  {name}")
        lines.append("  " + "-" * 68)
        lines.append(
            f"  {'band':>6}{'换手':>10}{'毛年化':>11}"
            f"{'成本':>10}{'净年化':>11}{'期数':>7}")
        for band, row in df.iterrows():
            lines.append(
                f"  {band:>6.2f}{row['turnover']:>9.1%}"
                f"{row['gross_ann']:>11.2%}{row['cost_ann']:>10.2%}"
                f"{row['net_ann']:>11.2%}{int(row['periods']):>7}")

    lines.append("")
    lines.append("  说明: 毛年化是多空收益的线性年化, 不含融券成本;")
    lines.append("        净年化 = 毛年化 - 每期成本均值 × 252/h。")
    return "\n".join(lines)
