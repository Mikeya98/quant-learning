"""
绩效指标计算 —— 评估策略好坏的标准工具箱。

核心指标：
- 总收益率 / 年化收益率：赚了多少
- 夏普比率：每承担一单位风险，获得了多少超额收益
- 最大回撤：最惨的时候亏了多少（以及持续了多久）
- 胜率 / 盈亏比：赢的次数多还是每次赢的金额大

这些指标共同构成策略的"体检报告"——没有单一指标能说明一切。
"""

import numpy as np
import pandas as pd


def total_return(equity_curve: pd.Series) -> float:
    """总收益率（%）"""
    return (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100


def annualized_return(equity_curve: pd.Series, trading_days: int = 252) -> float:
    """
    年化收益率（CAGR, %）

    公式: (final/initial)^(252/n_days) - 1
    """
    n = len(equity_curve)
    if n < 2:
        return 0.0
    return ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (trading_days / n) - 1) * 100


def daily_returns(equity_curve: pd.Series) -> pd.Series:
    """从净值曲线计算日收益率"""
    return equity_curve.pct_change().dropna()


def annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """年化波动率（%）"""
    return returns.std() * np.sqrt(trading_days) * 100


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.03,
                 trading_days: int = 252) -> float:
    """
    夏普比率

    公式: (年化收益 - 无风险利率) / 年化波动率
    无风险利率默认 3%（中国 10 年期国债近似）

    解读:
        > 1.0  不错
        > 2.0  优秀
        < 0    不如存银行
    """
    if len(equity_curve) < 2:
        return 0.0
    rets = daily_returns(equity_curve)
    ann_ret = annualized_return(equity_curve, trading_days) / 100
    ann_vol = annualized_volatility(rets, trading_days) / 100
    if ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def max_drawdown(equity_curve: pd.Series) -> dict:
    """
    最大回撤

    计算历史上从最高点到最低点的最大跌幅。

    Returns:
        dict with:
        - max_dd: 最大回撤百分比
        - peak_date: 最高点日期
        - trough_date: 最低点日期
        - duration: 回撤持续天数
    """
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100

    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin()

    # 找到峰值：回撤低点之前的最高点
    peak_idx = equity_curve[:trough_idx].idxmax()

    # 计算持续天数
    delta = trough_idx - peak_idx
    if hasattr(delta, 'days'):
        duration = delta.days
    else:
        duration = 0

    return {
        "max_dd": round(max_dd, 2),
        "peak_date": str(peak_idx)[:10],
        "trough_date": str(trough_idx)[:10],
        "duration_days": duration,
    }


def win_rate(trades_df: pd.DataFrame) -> float:
    """胜率（盈利交易数 / 总交易数, %）"""
    if len(trades_df) == 0:
        return 0.0
    return (trades_df["pnl"] > 0).sum() / len(trades_df) * 100


def profit_factor(trades_df: pd.DataFrame) -> float:
    """
    盈亏比（Profit Factor）

    公式: 总盈利 / 总亏损（取绝对值）
    > 1 表示盈利大于亏损
    """
    if len(trades_df) == 0:
        return 0.0
    gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def avg_trade_return(trades_df: pd.DataFrame) -> float:
    """平均每笔交易收益率（%）"""
    if len(trades_df) == 0:
        return 0.0
    return trades_df["pnl_pct"].mean()


def calmar_ratio(equity_curve: pd.Series, trading_days: int = 252) -> float:
    """
    Calmar 比率 = 年化收益率 / 最大回撤（取绝对值）

    衡量每承担一单位回撤风险获得的收益
    """
    ann_ret = annualized_return(equity_curve, trading_days)
    dd_info = max_drawdown(equity_curve)
    if dd_info["max_dd"] == 0:
        return 0.0
    return ann_ret / abs(dd_info["max_dd"])


def compute_all(result) -> dict:
    """
    一次性计算所有指标。

    Args:
        result: BacktestResult 对象（含 equity_df 和 trades_df）

    Returns:
        dict 包含所有指标
    """
    eq = result.equity_df
    if len(eq) == 0:
        return {}

    # 用日期索引构造净值序列
    equity = eq.set_index("date")["equity"]
    rets = daily_returns(equity)
    dd = max_drawdown(equity)

    return {
        "total_return_pct": round(total_return(equity), 2),
        "annual_return_pct": round(annualized_return(equity), 2),
        "annual_volatility_pct": round(annualized_volatility(rets), 2),
        "sharpe_ratio": round(sharpe_ratio(equity), 2),
        "max_drawdown_pct": dd["max_dd"],
        "max_dd_duration_days": dd["duration_days"],
        "calmar_ratio": round(calmar_ratio(equity), 2),
        "n_trades": len(result.trades_df),
        "win_rate_pct": round(win_rate(result.trades_df), 1),
        "profit_factor": round(profit_factor(result.trades_df), 2),
        "avg_trade_pnl_pct": round(avg_trade_return(result.trades_df), 2),
    }
