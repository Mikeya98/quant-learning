"""
技术指标计算 —— 纯函数集合，无状态、无副作用。

对于从 C 转 Python 的开发者：
- 所有函数都是向量化的（一次操作整个数组），不需要写 for 循环
- `series.rolling(n).mean()` 内部用 C 实现，比手写 for 循环快 100 倍
- 函数接收 pd.Series，返回 pd.Series 或 pd.DataFrame

指标说明：
- SMA: 简单移动平均，窗口内等权重
- EMA: 指数移动平均，近期数据权重更大
- MACD: 趋势跟踪指标，快慢 EMA 差值
- RSI: 相对强弱指数，衡量超买超卖，范围 0-100
- BB: 布林带，价格围绕均线的标准差通道
"""

import pandas as pd
import numpy as np


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    简单移动平均 (Simple Moving Average)

    公式: SMA = (P_t + P_{t-1} + ... + P_{t-period+1}) / period

    Args:
        series: 价格序列（通常是收盘价）
        period: 窗口大小

    Returns:
        SMA 序列，前 period-1 个值为 NaN
    """
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    指数移动平均 (Exponential Moving Average)

    公式: EMA_t = α × P_t + (1-α) × EMA_{t-1}
          其中 α = 2 / (period + 1)

    Args:
        series: 价格序列
        period: 平滑周期

    Returns:
        EMA 序列
    """
    return series.ewm(span=period, adjust=False).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD 指标 (Moving Average Convergence Divergence)

    公式:
        MACD 线 = EMA(fast) - EMA(slow)
        信号线 = EMA(MACD 线, signal)
        柱状图 = MACD 线 - 信号线

    Args:
        close:  收盘价序列
        fast:   快线周期（默认 12）
        slow:   慢线周期（默认 26）
        signal: 信号线周期（默认 9）

    Returns:
        DataFrame 包含列: dif, dea, hist
        - dif:  MACD 线（快慢 EMA 差值）
        - dea:  信号线（DIF 的 EMA）
        - hist: 柱状图（DIF - DEA，正值表示多头，负值表示空头）
    """
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    hist = dif - dea
    # hist * 2 是为了与常见行情软件显示一致（实际上 hist = 2*(dif-dea) 在某些实现中）
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist * 2}, index=close.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    相对强弱指数 (Relative Strength Index)

    公式: RSI = 100 - 100 / (1 + RS)
          其中 RS = 平均涨幅 / 平均跌幅（取绝对值）

    使用 Wilder 平滑方法（与主流行情软件一致）

    Args:
        close:  收盘价序列
        period: 计算周期（默认 14）

    Returns:
        RSI 序列，范围 0-100
        - RSI > 70: 超买区域，可能回调
        - RSI < 30: 超卖区域，可能反弹
    """
    delta = close.diff()
    gain = delta.clip(lower=0)   # 涨幅（正值保留，负值置 0）
    loss = -delta.clip(upper=0)  # 跌幅（取绝对值）

    # Wilder 平滑: 第一日用 SMA，之后用 EMA 方式递推
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    布林带 (Bollinger Bands)

    公式:
        中轨 = SMA(period)
        上轨 = 中轨 + num_std × 标准差
        下轨 = 中轨 - num_std × 标准差
        带宽 = (上轨 - 下轨) / 中轨 × 100

    Args:
        close:   收盘价序列
        period:  SMA 周期（默认 20）
        num_std: 标准差倍数（默认 2.0，约覆盖 95% 价格）

    Returns:
        DataFrame 包含列: upper, middle, lower, bandwidth
    """
    middle = sma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle * 100
    return pd.DataFrame(
        {"upper": upper, "middle": middle, "lower": lower, "bandwidth": bandwidth},
        index=close.index,
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    平均真实波幅 (Average True Range) —— 衡量市场波动性

    TR = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR = TR 的 EMA(period)

    Args:
        high:   最高价
        low:    最低价
        close:  收盘价
        period: ATR 周期

    Returns:
        ATR 序列
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_all(data: pd.DataFrame) -> pd.DataFrame:
    """
    一次性计算所有常用技术指标，添加到 DataFrame 中。

    Args:
        data: 含 close, high, low 列的 OHLCV DataFrame

    Returns:
        新增以下列的 DataFrame:
        - sma_5, sma_20, sma_60
        - ema_12, ema_26
        - macd_dif, macd_dea, macd_hist
        - rsi_14
        - bb_upper, bb_middle, bb_lower
    """
    df = data.copy()
    close = df["close"]

    df["sma_5"] = sma(close, 5)
    df["sma_20"] = sma(close, 20)
    df["sma_60"] = sma(close, 60)

    macd_df = macd(close)
    df["macd_dif"] = macd_df["dif"]
    df["macd_dea"] = macd_df["dea"]
    df["macd_hist"] = macd_df["hist"]

    df["rsi_14"] = rsi(close, 14)

    bb_df = bollinger_bands(close)
    df["bb_upper"] = bb_df["upper"]
    df["bb_middle"] = bb_df["middle"]
    df["bb_lower"] = bb_df["lower"]

    return df
