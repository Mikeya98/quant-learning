"""
可视化 —— 回测结果的图表展示。

包含：
- 净值曲线（策略 vs 买入持有基准）
- 回撤曲线
- 交易点位标注
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from pathlib import Path


# 图表保存目录
_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "charts"


def _ensure_output_dir():
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_equity_curve(
    equity_df: pd.DataFrame,
    data: pd.DataFrame = None,
    title: str = "Equity Curve",
    save: bool = True,
) -> str:
    """
    净值曲线图。

    上图为策略净值 vs 买入持有基准，下图为回撤。

    Args:
        equity_df: 回测输出的每日净值
        data:      原始 OHLCV 数据（计算买入持有基准用）
        title:     图表标题
        save:      是否保存到文件

    Returns:
        保存路径（如果 save=True），否则空字符串
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    equity = equity_df.set_index("date")["equity"]

    # 上方：净值曲线
    ax1.plot(equity.index, equity.values, color="#2c7fb8", linewidth=1.2, label="Strategy Equity")
    ax1.axhline(y=equity.iloc[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # 买入持有基准
    if data is not None:
        bh = data.set_index("date")["close"]
        bh = bh / bh.iloc[0] * equity.iloc[0]  # 归一化到同等初始资金
        ax1.plot(bh.index, bh.values, color="#d7191c", linewidth=0.8,
                 alpha=0.5, linestyle="--", label="Buy & Hold")

    ax1.set_ylabel("Equity (CNY)")
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # 标题含关键数字
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    dd = _max_drawdown_series(equity)
    ax1.set_title(f"{title}  |  Return: {total_ret:+.1f}%  |  Max DD: {dd:.1f}%",
                  fontsize=11, loc="left")

    # 下方：回撤曲线
    drawdown = _drawdown_series(equity)
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color="#d7191c", alpha=0.3, linewidth=0)
    ax2.plot(drawdown.index, drawdown.values, color="#d7191c", linewidth=0.8)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    fig.tight_layout()

    if save:
        _ensure_output_dir()
        path = _OUTPUT_DIR / "equity_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    plt.show()
    return ""


def plot_trade_signals(
    data: pd.DataFrame,
    trades_df: pd.DataFrame,
    title: str = "Trade Signals",
    save: bool = True,
) -> str:
    """
    价格走势图 + 买卖点标注。

    Args:
        data:      OHLCV 数据
        trades_df: 交易记录（含 entry_date, exit_date, entry_price, exit_price）
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    data_ts = data.set_index("date")
    ax.plot(data_ts.index, data_ts["close"], color="#333333", linewidth=0.8, alpha=0.7)

    # 标注买卖点
    for _, t in trades_df.iterrows():
        ax.scatter(t["entry_date"], t["entry_price"],
                   marker="^", color="#2ca02c", s=80, zorder=5)
        if pd.notna(t["exit_date"]):
            ax.scatter(t["exit_date"], t["exit_price"],
                       marker="v", color="#d7191c", s=80, zorder=5)

    # 用散点图例代替
    ax.scatter([], [], marker="^", color="#2ca02c", s=80, label="Buy")
    ax.scatter([], [], marker="v", color="#d7191c", s=80, label="Sell")

    ax.set_ylabel("Price (CNY)")
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if save:
        _ensure_output_dir()
        path = _OUTPUT_DIR / "trade_signals.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    plt.show()
    return ""


# ---- 内部辅助 ----

def _drawdown_series(equity: pd.Series) -> pd.Series:
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max * 100


def _max_drawdown_series(equity: pd.Series) -> float:
    return _drawdown_series(equity).min()
