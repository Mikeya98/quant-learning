"""
数据存取层 —— 负责 DataFrame 与 CSV 文件之间的读写。

设计原则：
- 所有数据以 CSV 格式存储在 data/ 目录下
- 文件名规范: {symbol}.csv（如 000001.csv）
- CSV 格式简单、可手动查看、零依赖
"""

import os
import pandas as pd


# 项目根目录（从本文件向上三级: storage.py -> data -> src -> quant_learning）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")


def _ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def save_csv(df: pd.DataFrame, symbol: str, processed: bool = False):
    """
    将 DataFrame 保存为 CSV 文件。

    Args:
        df: 要保存的 DataFrame
        symbol: 股票代码（如 "000001"），用作文件名
        processed: True 保存到 processed/ 目录，False 保存到 raw/ 目录
    """
    target_dir = PROCESSED_DIR if processed else RAW_DIR
    _ensure_dir(target_dir)
    filepath = os.path.join(target_dir, f"{symbol}.csv")
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"[storage] 已保存: {filepath} ({len(df)} 行)")


def load_csv(symbol: str, processed: bool = False) -> pd.DataFrame:
    """
    从 CSV 文件加载 DataFrame。

    Args:
        symbol: 股票代码（如 "000001"）
        processed: True 从 processed/ 加载，False 从 raw/ 加载

    Returns:
        pd.DataFrame，如果文件不存在则抛出 FileNotFoundError
    """
    target_dir = PROCESSED_DIR if processed else RAW_DIR
    filepath = os.path.join(target_dir, f"{symbol}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}\n请先运行: python main.py fetch {symbol}")
    df = pd.read_csv(filepath, encoding="utf-8-sig")
    # 如果日期列存在，自动转换为 datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    print(f"[storage] 已加载: {filepath} ({len(df)} 行)")
    return df


def list_stocks(processed: bool = False) -> list:
    """
    列出已下载的股票代码。

    Args:
        processed: True 查看 processed/ 目录，False 查看 raw/ 目录

    Returns:
        股票代码列表（不含 .csv 后缀）
    """
    target_dir = PROCESSED_DIR if processed else RAW_DIR
    if not os.path.exists(target_dir):
        return []
    files = [f for f in os.listdir(target_dir) if f.endswith(".csv")]
    return sorted([f.replace(".csv", "") for f in files])
