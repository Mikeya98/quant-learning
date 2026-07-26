"""
数据清洗层 —— 对原始数据进行标准化处理。

处理内容：
1. 停牌数据处理：识别并标记停牌日（volume=0 或价格连续不变）
2. 缺失值处理：前向填充价格，填充后仍缺失的行删除
3. 类型标准化：确保各列数据类型正确
4. 日期索引：设置 date 为索引（便于后续切片）

A股特点：
- 停牌期间无交易，volume=0，价格不变
- 涨跌停（±10% 普通股，±20% 科创/创业板）会导致极端涨跌幅
- T+1 交易制度，当天买入次日才能卖出
"""

import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame, fill_suspend: bool = True) -> pd.DataFrame:
    """
    清洗原始股票数据。

    Args:
        df: 原始 DataFrame（来自 fetcher）
        fill_suspend: 是否用前一日收盘价填充停牌日（默认 True，便于回测连续性）

    Returns:
        清洗后的 DataFrame
    """
    df = df.copy()

    # 1. 标记停牌日（volume 为 0 或 NaN 的日期）
    df["suspended"] = (df["volume"].isna()) | (df["volume"] <= 0)

    # 2. 缺失值处理
    # 价格列：前向填充（用前一日价格填充停牌/缺失日）
    price_cols = ["open", "high", "low", "close"]
    if fill_suspend:
        df[price_cols] = df[price_cols].ffill()
    # 成交量/成交额：缺失填 0
    df["volume"] = df["volume"].fillna(0)
    df["amount"] = df["amount"].fillna(0)

    # 填充后仍存在 NaN 的行（通常是数据开头几行）—— 删除
    before = len(df)
    df = df.dropna(subset=price_cols)
    after = len(df)
    if before != after:
        print(f"[cleaner] 删除了 {before - after} 行含有缺失价格的数据")

    # 3. 类型标准化
    df["volume"] = df["volume"].astype(np.int64)
    df["amount"] = df["amount"].astype(np.float64)
    for col in price_cols:
        df[col] = df[col].astype(np.float64)

    # 4. 按日期排序
    df = df.sort_values("date").reset_index(drop=True)

    # 5. 统计汇报
    suspended_days = df["suspended"].sum()
    total_days = len(df)
    print(f"[cleaner] 数据清洗完成: {total_days} 条, "
          f"停牌日 {suspended_days} 条 ({suspended_days/total_days*100:.1f}%)")

    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加收益率相关列（在回测时使用）。

    Args:
        df: 清洗后的 DataFrame

    Returns:
        添加了以下新列的 DataFrame:
        - ret:         日收益率（close / close.shift(1) - 1）
        - ret_log:     对数收益率（log(close / close.shift(1))）
        - cum_ret:     累计收益率（买入持有基准）
    """
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["ret_log"] = np.log(df["close"] / df["close"].shift(1))
    df["cum_ret"] = (1 + df["ret"].fillna(0)).cumprod()
    return df
