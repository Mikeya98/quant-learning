"""
组合回测可视化 —— 净值、回撤、仓位权重、相关矩阵。

和 visualize.py 的区别: 这里的输入是 PortfolioResult,
净值要和等权买入持有基准一起画, 另外还要看仓位怎么变化。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "charts"


def _ensure_output_dir():
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _drawdown_series(equity):
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max * 100


def plot_portfolio(result, save: bool = True):
    """
    组合净值 vs 等权基准 + 回撤。

    Returns:
        list[str] 保存的文件路径
    """
    eq = result.equity_df.set_index("date")
    equity = eq["equity"]
    benchmark = eq["benchmark"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax1.plot(equity.index, equity.values, color="#2c7fb8",
             linewidth=1.2, label="Portfolio")
    ax1.plot(benchmark.index, benchmark.values, color="#d7191c",
             linewidth=0.9, linestyle="--", alpha=0.7,
             label="Equal-weight buy & hold")
    ax1.axhline(y=equity.iloc[0], color="gray", linestyle="--",
                linewidth=0.7, alpha=0.5)

    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    bench_ret = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100
    max_dd = _drawdown_series(equity).min()
    ax1.set_title(
        f"{result.config.get('strategy', '')} | "
        f"Portfolio {total_ret:+.1f}% vs Benchmark {bench_ret:+.1f}% | "
        f"Max DD {max_dd:.1f}%",
        fontsize=11, loc="left",
    )
    ax1.set_ylabel("Equity (CNY)")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    dd = _drawdown_series(equity)
    ax2.fill_between(dd.index, dd.values, 0, color="#d7191c",
                     alpha=0.3, linewidth=0)
    ax2.plot(dd.index, dd.values, color="#d7191c", linewidth=0.8)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.tight_layout()
    paths = []

    if save:
        _ensure_output_dir()
        p1 = _OUTPUT_DIR / "portfolio_equity.png"
        fig.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(p1))
    else:
        plt.show()

    # ---- 权重堆叠图 ----
    wdf = result.weights_df.set_index("date")
    symbols = [c for c in wdf.columns if c != "date" and wdf[c].sum() > 0]

    fig2, ax = plt.subplots(figsize=(14, 5))
    if symbols:
        ax.stackplot(wdf.index, [wdf[s].values for s in symbols],
                     labels=symbols, alpha=0.85)
        ax.legend(loc="upper left", ncol=min(4, len(symbols)),
                  fontsize=8, framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Weight")
    ax.set_title("Portfolio weights (cash is the blank area)",
                 fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    fig2.tight_layout()
    if save:
        p2 = _OUTPUT_DIR / "portfolio_weights.png"
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        paths.append(str(p2))
    else:
        plt.show()

    return paths


def plot_correlation(result, save: bool = True):
    """日收益率相关矩阵热力图"""
    corr = result.correlation
    if corr.empty:
        return ""

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-1, vmax=1)

    labels = list(corr.columns)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Daily return correlation", fontsize=11, loc="left")

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(v) < 0.6 else "white")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save:
        _ensure_output_dir()
        path = _OUTPUT_DIR / "portfolio_correlation.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    plt.show()
    return ""
