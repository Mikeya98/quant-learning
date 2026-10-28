"""
股票元数据 —— 行业分类与市值。

因子中性化要控的两个经典暴露：

    行业    同行业股票共享景气度。一个"动量因子"如果只是把某几个
            行业选出来了，IC 看起来不错，实际上赌的是行业轮动。
    市值    A 股小市值效应很强。很多因子（换手率、波动率、反转）
            和市值高度相关，本质是在赌大小盘。

行业是分类变量，从 data/meta/industry.csv 读；市值由日线反推：

    流通市值 ≈ 成交额 / 换手率

不用行情快照接口里的"当前总市值"：那是今天的数字，拿它去中性化
2019 年的因子等于用了未来信息。五年里涨跌、送转、增发都会让市值
排名完全变样。而成交额和换手率是当天就有的，算出来的是当天的
流通市值，时点上是对的。

数据约定（和回测框架一致）：
    DataFrame index = 交易日，columns = 股票代码
"""

import os

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

META_DIR = os.path.join(_PROJECT_ROOT, "data", "meta")
INDUSTRY_FILE = os.path.join(META_DIR, "industry.csv")

# 市值合理区间（元）。超出这个范围的 amount/turnover 基本是脏数据：
# 换手率字段偶尔为 0 或者单位不对，会让市值算成天文数字。
MKT_CAP_MIN = 1e8       # 1 亿
MKT_CAP_MAX = 1e13      # 10 万亿


# ===================================================================
# 行业分类
# ===================================================================

def load_industry(path: str = None) -> pd.Series:
    """
    读取行业分类表。

    CSV 格式（见 data/meta/industry.csv）：
        symbol,name,industry
        600519,贵州茅台,食品饮料

    Returns
    -------
    pd.Series，index=symbol，值=行业名。文件不存在时返回空 Series。
    """
    path = path or INDUSTRY_FILE
    if not os.path.exists(path):
        return pd.Series(dtype=object, name="industry")

    df = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    df["symbol"] = df["symbol"].str.zfill(6)
    s = df.set_index("symbol")["industry"].dropna()
    s.name = "industry"
    return s


def save_industry(industry: pd.Series, path: str = None,
                  names: pd.Series = None) -> str:
    """把 symbol → 行业 的映射写回 CSV（覆盖）"""
    path = path or INDUSTRY_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)

    idx = pd.Index(industry.index).astype(str).str.zfill(6)
    df = pd.DataFrame({"symbol": idx})
    if names is not None:
        df["name"] = [names.get(sym, "") for sym in idx]
    else:
        df["name"] = ""
    df["industry"] = industry.values
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def industry_wide(industry: pd.Series, index, columns) -> pd.DataFrame:
    """
    把 symbol → 行业 的 Series 铺成和因子一样的宽表。

    行业在样本期内一般不变化，所以每一行都是同一份映射；铺成宽表
    只是为了和因子宽表对齐（对齐这件事本身交给 reindex）。
    """
    row = industry.reindex(pd.Index(columns, dtype=object))
    return pd.DataFrame([row.values] * len(index),
                        index=index, columns=columns)


# ===================================================================
# 市值
# ===================================================================

def float_mktcap(amount: pd.DataFrame, turnover: pd.DataFrame) -> pd.DataFrame:
    """
    由成交额和换手率反推流通市值。

        换手率 = 成交量 / 流通股本
        成交额 ≈ 成交量 × 均价
        →  流通市值 = 流通股本 × 现价 ≈ 成交额 / 换手率

    Parameters
    ----------
    amount   : 成交额（元）
    turnover : 换手率（小数，0.0031 表示 0.31%）

    Returns
    -------
    流通市值宽表，单位元。字段缺失或数值离谱的格子为 NaN。
    """
    amount, turnover = amount.align(turnover, join="inner")

    turnover = turnover.where(turnover > 0)
    cap = amount / turnover

    # 停牌日 amount=0、换手率=0，算出来是 0 或 inf，都要剔掉
    cap = cap.where((cap >= MKT_CAP_MIN) & (cap <= MKT_CAP_MAX))
    return cap


def log_size(mktcap: pd.DataFrame) -> pd.DataFrame:
    """市值取自然对数 —— 市值分布右偏严重，取对数之后才适合做线性回归"""
    return np.log(mktcap.where(mktcap > 0))
