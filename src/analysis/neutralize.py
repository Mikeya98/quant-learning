"""
因子中性化 —— 去掉行业和市值暴露。

一个因子有 IC，不代表它发现了"新东西"。最常见的两种情况：

    行业暴露   因子值在某个行业系统性偏高。比如 2020 年的动量因子，
               选出来一大半是新能源 —— 它的 IC 来自"新能源涨了"，
               而不是"动量有效"。

    市值暴露   因子和市值高度相关。小市值股票波动大、换手高，
               很多因子天然选出小盘股，赚的是小市值溢价，
               换成大盘股样本就失效。

中性化的做法就是把这些"已知的、你不想要的"暴露当自变量，把因子
当因变量做回归，取残差：

    因子 = a + b1·行业哑变量 + b2·ln(市值) + ε

    ε 就是去掉这些暴露之后剩下的部分，也就是"净因子"。

    行业是分类变量，用"行业内去均值"实现（等价于行业哑变量回归）；
    市值是连续变量，用逐日横截面回归。

注意两点：

1. 每次中性化都必须逐日做横截面。把整段样本放在一起回归会把
   时间维度的信息（比如整体牛市）混进来，等于作弊。

2. 中性化不是万能的。它去掉了暴露，也去掉了因子的一部分预测力；
   如果中性化之后 IC 掉到 0，说明这个因子的收益本来就来自
   行业/市值，而不是选股能力 —— 这个结论本身就是有价值的。

数据约定：index=交易日，columns=股票代码。
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .factor_research import zscore_cross_section

__all__ = [
    "winsorize_cross_section",
    "mad_winsorize",
    "industry_neutralize",
    "size_neutralize",
    "residualize",
    "neutralize_factor",
    "industry_exposure",
    "size_exposure",
]


# ===================================================================
# 第一步：去极值
# ===================================================================

def winsorize_cross_section(
    df: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """
    每日横截面分位数缩尾。

    把低于 lower 分位的值拉回 lower 分位，高于 upper 分位的拉回
    upper 分位。A 股里一个因子经常被一两只极端股票带偏（比如
    刚重组的、连续涨停的），缩尾比直接删除稳健。

    Parameters
    ----------
    lower, upper : 分位点 (0, 1)，例如 (0.01, 0.99)
    """
    if not (0.0 <= lower < upper <= 1.0):
        raise ValueError("need 0 <= lower < upper <= 1")

    def _one_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if len(valid) < 3:
            return row
        lo = valid.quantile(lower)
        hi = valid.quantile(upper)
        return row.clip(lower=lo, upper=hi)

    return df.apply(_one_row, axis=1)


def mad_winsorize(df: pd.DataFrame, n: float = 3.0) -> pd.DataFrame:
    """
    用中位数绝对偏差 (MAD) 去极值。

    比固定分位点更适应分布变化：正常分布下 3 倍 MAD 大约对应
    2 倍标准差，但遇到厚尾或者一半样本都是同一个值时，
    MAD 不会被拉偏。
    """
    med = df.median(axis=1)
    mad = (df.sub(med, axis=0)).abs().median(axis=1)
    # 1.4826: MAD → 标准差的无偏换算系数（正态分布下）
    scale = (1.4826 * mad).replace(0, np.nan)

    lower = med - n * scale
    upper = med + n * scale
    return df.clip(lower=lower, upper=upper, axis=0)


# ===================================================================
# 行业中性化
# ===================================================================

def _industry_map(industry, columns: pd.Index) -> pd.Series:
    """
    把行业信息统一成 "每列一个行业" 的 Series。

    支持两种输入:
        Series           symbol → 行业
        DataFrame        日期 × symbol 的宽表（取第一行）
    """
    if isinstance(industry, pd.DataFrame):
        if industry.empty:
            return pd.Series(index=columns, dtype=object)
        industry = industry.iloc[0]

    s = pd.Series(industry).reindex(pd.Index(columns, dtype=object))
    # 没有行业信息的股票归到一组，不要直接丢弃 —— 丢弃会让样本
    # 每天不一样，IC 序列不可比。
    return s.fillna("__unknown__")


def industry_neutralize(
    factor: pd.DataFrame,
    industry,
    min_group: int = 2,
    scale: bool = False,
) -> pd.DataFrame:
    """
    行业中性化：每只股票的因子值减去同行业当天的均值。

    Parameters
    ----------
    factor    : 因子宽表
    industry  : symbol → 行业 (Series) 或行业宽表 (DataFrame)
    min_group : 行业内至少几只股票才做去均值。只有一只的行业
                减完就恒等于 0，等于把这只股票的信息抹掉了，
                所以样本太少的行业保持原值。
    scale     : True 时再除以行业内的标准差（行业标准化）。
                一般只要去均值就够了；缩放会让小行业里的一只
                异常股票被放大。

    Returns
    -------
    同形状的 DataFrame
    """
    ind = _industry_map(industry, factor.columns)

    # 按行业把列分组，对每一行做组内去均值。
    # groupby 作用在转置后的表上: 行=股票, 列=日期, 这样
    # transform("mean") 天然就是"同一行业、同一天"的均值。
    grouped = factor.T.groupby(ind)

    sizes = grouped.size()
    small = set(sizes[sizes < min_group].index)

    mean = grouped.transform("mean").T
    out = factor - mean

    if scale:
        std = grouped.transform("std").T
        out = out / std.replace(0, np.nan)

    # 样本太少的行业: 还原成原始值
    if small:
        keep = ind.isin(small)
        if keep.any():
            cols = list(ind.index[keep])
            out.loc[:, cols] = factor.loc[:, cols]

    return out


# ===================================================================
# 市值中性化
# ===================================================================

def _residualize_row(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    对一行做 OLS，返回残差。NaN 的位置保持 NaN。

    X 已经包含常数项。样本数必须大于回归元个数，否则残差恒为 0，
    没有意义 —— 这种情况直接返回全 NaN，让调用方按"这天算不出来"处理。
    """
    res = np.full_like(y, np.nan, dtype=float)

    mask = ~np.isnan(y) & np.all(~np.isnan(X), axis=1)
    n = int(mask.sum())
    if n <= X.shape[1]:
        return res

    yy = y[mask]
    XX = X[mask]

    try:
        beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    except np.linalg.LinAlgError:
        return res

    res[mask] = yy - XX @ beta
    return res


def residualize(
    factor: pd.DataFrame,
    exposures: Sequence[pd.DataFrame],
    add_constant: bool = True,
) -> pd.DataFrame:
    """
    多元回归取残差：factor = a + Σ b_i·exposure_i + ε

    逐日横截面回归。exposures 里每个都是和 factor 同形状的宽表
    （例如行业哑变量、ln 市值）。

    多个暴露之间可能共线（行业哑变量之和恒为 1），lstsq 用最小
    二乘的最小范数解能处理，残差仍然唯一。
    """
    if not exposures:
        return factor.copy()

    factor, _ = factor.align(exposures[0], join="inner")

    X = np.empty((len(factor.columns), len(exposures) + int(add_constant)))
    out = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)

    exps = [m.reindex(index=factor.index, columns=factor.columns)
            for m in exposures]

    for dt in factor.index:
        y = factor.loc[dt].to_numpy(dtype=float)
        for j, e in enumerate(exps):
            X[:, j] = e.loc[dt].to_numpy(dtype=float)
        if add_constant:
            X[:, -1] = 1.0

        out.loc[dt] = _residualize_row(y, X)

    return out


def size_neutralize(
    factor: pd.DataFrame,
    log_mktcap: pd.DataFrame,
) -> pd.DataFrame:
    """
    市值中性化：逐日把因子对 ln(流通市值) 做一元回归，取残差。

    这就是 Fama-MacBeth 回归里最基础的一步控制变量。做完之后，
    因子和市值的横截面相关系数应该接近 0。
    """
    factor, log_mktcap = factor.align(log_mktcap, join="inner")
    return residualize(factor, [log_mktcap], add_constant=True)


# ===================================================================
# 组合流程
# ===================================================================

def neutralize_factor(
    factor: pd.DataFrame,
    industry=None,
    log_mktcap: Optional[pd.DataFrame] = None,
    winsorize: Optional[Tuple[float, float]] = (0.01, 0.99),
    standardize: bool = True,
    industry_scale: bool = False,
) -> pd.DataFrame:
    """
    完整的中性化流程。

        缩尾 → 行业去均值 → 市值回归取残差 → 横截面 z-score

    顺序有讲究:
      - 先缩尾: 极端值会把市值回归的斜率带偏
      - 再行业: 行业去均值是减法, 和市值回归谁先谁后差别不大,
        但先去行业可以让市值回归的残差更干净
      - 最后标准化: 中性化之后量纲变了, 合成多因子之前必须重新
        z-score, 否则不同因子的权重没有可比性

    Parameters
    ----------
    industry    : symbol → 行业，或行业宽表。None 表示不做行业中性化
    log_mktcap  : ln(流通市值) 宽表。None 表示不做市值中性化
    winsorize   : (lower, upper) 分位点；None 表示不缩尾
    standardize : 最后是否做横截面 z-score
    """
    out = factor.copy()

    if winsorize is not None:
        out = winsorize_cross_section(out, *winsorize)
    else:
        out = out.where(np.isfinite(out))

    if industry is not None:
        out = industry_neutralize(out, industry, scale=industry_scale)

    if log_mktcap is not None:
        out = size_neutralize(out, log_mktcap)

    if standardize:
        z = zscore_cross_section(out)
        # zscore 会把缺失值填成 0（对合成打分是把"没意见"当中性），
        # 但因子研究里 0 是一个有意义的取值。原因子没有数据的位置
        # 必须保持 NaN，否则等于凭空造了一个"中性"样本。
        out = z.where(factor.notna())

    return out


# ===================================================================
# 暴露检查
# ===================================================================

def industry_exposure(factor: pd.DataFrame, industry) -> pd.Series:
    """
    每个行业的平均因子值（先按日 z-score 再取均值）。

    中性化之后这张表应该接近 0；如果某个行业还是明显不为 0，
    说明那一组样本太少（或者行业字段缺失）。
    """
    ind = _industry_map(industry, factor.columns)
    z = zscore_cross_section(factor)
    per_day = z.T.groupby(ind).mean().T
    return per_day.mean()


def size_exposure(factor: pd.DataFrame,
                  log_mktcap: pd.DataFrame) -> float:
    """
    因子和 ln(市值) 的横截面相关系数的日均值。

    中性化之前通常是 -0.3 ~ -0.6（小盘股因子值高），
    中性化之后应该接近 0。
    """
    factor, log_mktcap = factor.align(log_mktcap, join="inner")

    vals = []
    for dt in factor.index:
        f = factor.loc[dt]
        s = log_mktcap.loc[dt]
        mask = f.notna() & s.notna()
        if mask.sum() < 5:
            continue
        fv = f[mask].astype(float)
        sv = s[mask].astype(float)
        if fv.nunique() < 2 or sv.nunique() < 2:
            continue
        c = fv.corr(sv)
        if pd.notna(c):
            vals.append(c)

    return float(np.mean(vals)) if vals else float("nan")
