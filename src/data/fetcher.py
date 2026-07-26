"""
数据获取层 —— 通过 akshare 下载 A 股历史日线数据。

akshare 是 Python 开源金融数据接口库，封装了东方财富、新浪财经等数据源。
我们用 stock_zh_a_hist() 获取个股历史日线数据（OHLCV）。

关键概念：
- 前复权(qfq): 以当前价格为基准，向后调整历史价格，适合回测
- 后复权(hfq): 以历史价格为基准，向前调整当前价格
- 不复权: 原始价格，含分红送股缺口，不适合回测
"""

import time
import akshare as ak
import pandas as pd


def fetch_stock_daily(
    symbol: str,
    start_date: str = "20100101",
    end_date: str = "20241231",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    下载 A 股个股历史日线数据。

    Args:
        symbol: 股票代码，6位数字字符串，如 "000001"（平安银行）、"600519"（贵州茅台）
        start_date: 起始日期，格式 "YYYYMMDD"
        end_date: 结束日期，格式 "YYYYMMDD"
        adjust: 复权方式 - "qfq" 前复权, "hfq" 后复权, "" 不复权

    Returns:
        pd.DataFrame，包含以下列：
        - date:       日期
        - open:       开盘价
        - high:       最高价
        - low:        最低价
        - close:      收盘价
        - volume:     成交量（手）
        - amount:     成交额（元）
        - amplitude:  振幅（%）
        - pct_change: 涨跌幅（%）
        - change:     涨跌额（元）
        - turnover:   换手率（%）

    Example:
        >>> df = fetch_stock_daily("000001", "20230101", "20231231")
        >>> print(df.head())
    """
    print(f"[fetcher] 正在下载 {symbol} 数据 ({start_date} ~ {end_date}, {adjust or '不复权'})...")

    # 网络请求重试（东方财富 API 偶发断连，重试 3 次）
    max_retries = 3
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust,
            )
            break  # 成功，跳出重试循环
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait = 2 ** attempt  # 指数退避: 2s, 4s, 8s
                print(f"[fetcher] 第 {attempt} 次尝试失败，{wait}s 后重试...")
                time.sleep(wait)
    else:
        # 所有重试均失败
        raise RuntimeError(
            f"下载 {symbol} 数据失败（重试 {max_retries} 次后仍失败）: {last_error}\n"
            f"请检查: 1) 网络连接 2) 股票代码是否正确 3) akshare 版本"
        )

    if df is None or len(df) == 0:
        raise ValueError(f"{symbol} 在指定日期范围内无数据")

    # akshare 返回的列名是中文，重命名为英文
    column_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    # 只保留存在的列（不同 akshare 版本可能略有差异）
    rename_map = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # 确保日期列正确
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # 按日期升序排列（回测需要时间顺序）
    df = df.sort_values("date").reset_index(drop=True)

    # 基本数据校验
    _validate(df, symbol)

    print(f"[fetcher] 下载完成: {symbol}, 共 {len(df)} 条记录, "
          f"日期范围 {df['date'].min().date()} ~ {df['date'].max().date()}")

    return df


def _validate(df: pd.DataFrame, symbol: str):
    """基本数据质量校验"""
    required_cols = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol} 数据缺少必要列: {missing}")

    # 检查价格合理性（A股价格至少 > 0.01 元）
    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            raise ValueError(f"{symbol} {col} 列存在非正值")

    # 检查 high >= low
    if (df["high"] < df["low"]).any():
        raise ValueError(f"{symbol} 存在 high < low 的异常行")

    # 检查日期是否重复
    if df["date"].duplicated().any():
        dup_dates = df["date"][df["date"].duplicated()].tolist()
        raise ValueError(f"{symbol} 存在重复日期: {dup_dates}")
