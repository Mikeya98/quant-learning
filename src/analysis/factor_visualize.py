"""
因子研究可视化 —— IC 序列和分层净值。

两张图回答两个问题:
    1. 这个因子每天打分准不准? (IC 序列 + 滚动均值)
    2. 按它分组买, 收益是不是单调的? (分层累计收益)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "charts"


def _ensure_output_dir():
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_ic(ic_result, holdings: int = 5, rolling: int = 20,
            title: str = "Factor IC", save: bool = True):
    """
    画 IC 时间序列和滚动均值。

    IC 单日波动很大，只看均值看不出稳定性；滚动 20 日均线在 0 以上
    持续越久，说明因子越可靠。
    """
    _ensure_output_dir()

    ic = ic_result.ic_series
    if ic.empty:
        return []

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = ["#d62728" if v < 0 else "#2ca02c" for v in ic.values]
    ax.bar(ic.index, ic.values, width=1.0, color=colors, alpha=0.45,
           label="daily IC")

    roll = ic.rolling(rolling, min_periods=max(2, rolling // 2)).mean()
    ax.plot(roll.index, roll.values, color="#1f77b4", linewidth=1.8,
            label=f"{rolling}d rolling mean")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(ic_result.mean_ic, color="#ff7f0e", linestyle="--",
               linewidth=1.2, label=f"mean IC = {ic_result.mean_ic:.4f}")

    ax.set_title(f"{title}  (holding {holdings}d, "
                 f"mean={ic_result.mean_ic:.4f}, ICIR={ic_result.icir:.3f})")
    ax.set_ylabel("IC")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    paths = []
    if save:
        p = _OUTPUT_DIR / f"factor_ic_{title.replace(' ', '_')}.png"
        fig.savefig(p, dpi=130)
        paths.append(str(p))
    plt.close(fig)
    return paths


def plot_quantiles(qret, holdings: int = 5, title: str = "Factor Quantiles",
                   save: bool = True):
    """
    分层累计收益。Q1 = 因子值最小，Qn = 最大。

    看两件事: 曲线是否单调, 以及 Qn - Q1 的价差（多空组合）。
    """
    _ensure_output_dir()

    if qret is None or qret.empty:
        return []

    fig, ax = plt.subplots(figsize=(14, 6))

    cmap = plt.get_cmap("RdYlGn")
    n = qret.shape[1]
    for i, col in enumerate(qret.columns):
        cum = (1 + qret[col].fillna(0)).cumprod()
        ax.plot(cum.index, cum.values,
                color=cmap(i / max(1, n - 1)), linewidth=1.4, label=col)

    # 多空组合: 每期 Qn 收益 - Q1 收益
    if n >= 2:
        long_short = qret.iloc[:, -1] - qret.iloc[:, 0]
        ls_cum = (1 + long_short.fillna(0)).cumprod()
        ax.plot(ls_cum.index, ls_cum.values, color="black", linewidth=1.8,
                linestyle="--", label=f"{qret.columns[-1]}-{qret.columns[0]}")

    ax.set_title(f"{title} (cumulative, holding {holdings}d)")
    ax.set_ylabel("growth of 1")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    paths = []
    if save:
        p = _OUTPUT_DIR / f"factor_quantiles_{title.replace(' ', '_')}.png"
        fig.savefig(p, dpi=130)
        paths.append(str(p))
    plt.close(fig)
    return paths
