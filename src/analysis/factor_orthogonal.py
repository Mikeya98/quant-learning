"""
因子正交化 —— 让多个因子在合成之前先互不解释。

多因子合成最容易被忽略的问题是因子之间高度相关。动量 20 日和动量 60
日、换手率和成交额、波动率和反转，横截面上经常是一回事。等权相加
等于把同一个信号算了两三遍，权重表看起来是三个因子，实际暴露只有
一个。

正交化的做法是逐日横截面回归，按给定顺序把后面的因子对前面的因子
（已经取过残差的版本）回归，只留下残差：

    f1' = f1
    f2' = f2 - proj(f2 | f1')
    f3' = f3 - proj(f3 | f1', f2')

残差和前面所有因子在横截面上正交，因此合成时每个因子贡献的是
"别人没解释掉的那部分"。顺序有意义：排在前面的因子保留原样，后面
的因子被剥离，所以重要或先验更可靠的因子放前面。

注意两点：

1. 每个交易日单独回归，不能把时间维度混进来。和中性化一样，
   一旦跨日回归，残差里就会带进市场整体的时序信息。

2. 正交化不创造信息。如果两个因子本质上相同，第二个被剥完之后
   只剩噪声，它的 IC 也会掉到 0 —— 这个结果说明它本来就不该单独
   占一份权重。
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .neutralize import neutralize_factor

__all__ = [
    "orthogonalize_factors",
    "mean_pairwise_corr",
    "prepare_factors",
]


def _residualize_on(
    target: pd.DataFrame,
    regressors: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """逐日把 target 对 regressors 做带截距 OLS，返回残差。

    只使用当天所有列都有效的样本；缺值的股票在残差里保持 NaN，
    不会因为"没数据"被塞进一个 0 残差。
    """
    if not regressors:
        return target.copy()

    index = target.index
    columns = target.columns
    aligned = [r.reindex(index=index, columns=columns) for r in regressors]
    out = pd.DataFrame(np.nan, index=index, columns=columns)

    for dt in index:
        y = target.loc[dt].to_numpy(dtype=float)
        X = np.column_stack([
            r.loc[dt].to_numpy(dtype=float) for r in aligned
        ] + [np.ones(len(columns))])

        mask = ~np.isnan(y) & np.all(~np.isnan(X), axis=1)
        n = int(mask.sum())
        if n <= X.shape[1]:
            continue

        try:
            beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        except np.linalg.LinAlgError:
            continue

        row = out.loc[dt].to_numpy(dtype=float, copy=True)
        row[mask] = y[mask] - X[mask] @ beta
        out.loc[dt] = row

    return out


def orthogonalize_factors(
    factors: Dict[str, pd.DataFrame],
    order: Optional[Iterable[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """按顺序逐日正交化一组因子。

    Parameters
    ----------
    factors : 因子名 → 宽表 (index=日期, columns=代码)
    order   : 处理顺序。默认用字典的插入顺序；排在前面的因子原样保留。

    Returns
    -------
    同形状的字典，键的顺序不变。第一个因子是原表的副本（不做去均值，
    合成前还会做横截面 z-score），后面的因子是相对前面所有残差的
    回归残差。
    """
    if not factors:
        return {}

    names = list(order) if order is not None else list(factors.keys())

    unknown = [n for n in names if n not in factors]
    if unknown:
        raise ValueError(f"未知因子: {', '.join(unknown)}")
    missing = [n for n in factors if n not in names]
    if missing:
        # 允许调用方只挑一部分，但漏掉的因子不应被静默丢弃
        raise ValueError(f"order 缺少因子: {', '.join(missing)}")

    # 统一索引和列，后面逐日回归不用反复对齐
    base = factors[names[0]]
    index = base.index
    columns = base.columns

    out: Dict[str, pd.DataFrame] = {}
    done: list[pd.DataFrame] = []

    for name in names:
        f = factors[name].reindex(index=index, columns=columns)
        if not done:
            resid = f.copy()
        else:
            resid = _residualize_on(f, done)
        out[name] = resid
        done.append(resid)

    return out


def mean_pairwise_corr(
    factors: Dict[str, pd.DataFrame],
) -> Dict[tuple, float]:
    """逐日横截面 Pearson 相关的平均值，用来看因子之间有多像。

    只统计当天两只因子都有效、且样本数 >= 3 的交易日。
    """
    names = list(factors.keys())
    result: Dict[tuple, float] = {}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = factors[names[i]]
            b = factors[names[j]]
            a, b = a.align(b, join="inner")

            values = []
            for dt in a.index:
                x = a.loc[dt].to_numpy(dtype=float)
                y = b.loc[dt].to_numpy(dtype=float)
                mask = ~np.isnan(x) & ~np.isnan(y)
                if mask.sum() < 3:
                    continue
                xs = x[mask]
                ys = y[mask]
                if np.std(xs) == 0 or np.std(ys) == 0:
                    continue
                values.append(float(np.corrcoef(xs, ys)[0, 1]))

            key = (names[i], names[j])
            result[key] = float(np.mean(values)) if values else float("nan")

    return result


def prepare_factors(
    library: Dict[str, pd.DataFrame],
    names: Sequence[str],
    *,
    winsorize: bool = False,
    industry=None,
    log_mktcap: Optional[pd.DataFrame] = None,
    orthogonalize: bool = False,
) -> Dict[str, pd.DataFrame]:
    """选股流程里的因子预处理。

    顺序和 neutralize_factor 保持一致：

        缩尾 → 行业去均值 → 市值回归取残差 → 横截面 z-score → 正交化

    标准化的位置值得说明：先 z-score 再正交化，回归的数值条件更好，
    而且正交化之后残差的尺度可能变化，合成时 combine_factors 还会
    再做一次 z-score，不影响权重。
    """
    chosen = {n: library[n] for n in names}

    need_clean = winsorize or industry is not None or log_mktcap is not None
    if need_clean:
        chosen = {
            n: neutralize_factor(
                df,
                industry=industry,
                log_mktcap=log_mktcap,
                winsorize=(0.01, 0.99) if winsorize else None,
                standardize=True,
            )
            for n, df in chosen.items()
        }

    if orthogonalize:
        chosen = orthogonalize_factors(chosen, order=list(names))

    return chosen
