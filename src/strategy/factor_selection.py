"""
多因子选股 —— 把因子研究的结果接进组合回测。

前面的 factor_research 只回答"这个因子有没有预测能力"，得到的是
IC、分层收益这类统计量。这一步把结论变成仓位：

    每个调仓日 → 横截面打分 → 取分数最高的 top_n 只 → 等权持有

几个容易踩的坑，这里都显式处理：

1. 未来信息
   因子只用 t 日及以前的收盘价计算，打分也在 t 日收盘后完成。
   组合引擎的 next_open 模式会用 t-1 的信号在 t 日开盘成交，
   正好对应"今天收盘出信号、明天开盘下单"的现实流程。

2. 因子量纲
   动量是百分比、波动率是标准差、换手率是倍数，直接相加没有意义。
   combine_factors 会先做横截面 z-score，再等权/加权合成。

3. 数据不足
   动量要 20 天历史，刚上市或长期停牌的股票分数是 NaN。
   这些标的不参与排序，不会被"0 分"混进组合。

4. 持仓连续性
   相邻两个调仓日之间不换股。只有调仓日才重新排序，其余日期
   沿用上一次的选择，避免每天按噪声换手。
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from ..analysis.factor_orthogonal import prepare_factors
from ..analysis.factor_research import (
    combine_factors,
    compute_forward_returns,
    compute_ic,
    make_factors,
    zscore_cross_section,
)

__all__ = [
    "DEFAULT_FACTORS",
    "composite_score",
    "rolling_ic_weights",
    "ic_weighted_score",
    "select_top",
    "signals_from_selection",
    "selection_turnover",
]

# 默认使用的因子：中期动量 + 短期反转 + 低波动
DEFAULT_FACTORS = ("momentum_20", "reversal_5", "vol_20")


def composite_score(
    close: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.DataFrame] = None,
    factors: Optional[Iterable[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    *,
    winsorize: bool = False,
    industry=None,
    log_mktcap: Optional[pd.DataFrame] = None,
    orthogonalize: bool = False,
) -> pd.DataFrame:
    """构造多因子合成分数 (宽表: index=日期, columns=代码)。

    因子候选取自 factor_research.make_factors；不在候选里的名字直接报错，
    免得手误时静默变成一个空组合。

    v0.19 起可以先做预处理再合成：
      winsorize       每日分位缩尾，压掉极端值
      industry        行业宽表/Series，传入后逐日行业内去均值
      log_mktcap      ln(流通市值) 宽表，传入后逐日回归取残差
      orthogonalize   按 factors 的顺序逐日正交化，后面的因子剥离前面
                      已经解释掉的部分
    """
    library = make_factors(close, volume, turnover)
    names = list(factors) if factors else [n for n in DEFAULT_FACTORS if n in library]

    unknown = [n for n in names if n not in library]
    if unknown:
        raise ValueError(
            f"未知因子: {', '.join(unknown)}；可用: {', '.join(library.keys())}"
        )
    if not names:
        raise ValueError("没有可用因子")

    chosen = prepare_factors(
        library, names,
        winsorize=winsorize,
        industry=industry,
        log_mktcap=log_mktcap,
        orthogonalize=orthogonalize,
    )

    # combine_factors 会把 NaN 填成 0 分，直接拿来排序会让"没有数据"
    # 的股票混进中位。先记下每个位置是否所有因子都有值，最后再掩回去。
    valid = pd.DataFrame(True, index=close.index, columns=close.columns)
    for df in chosen.values():
        valid &= df.notna()

    # 只给选中的因子权重，避免调用方把旧权重带进来
    if weights is not None:
        weights = {n: weights.get(n, 0.0) for n in names}

    score = combine_factors(chosen, weights=weights)
    return score.where(valid)


def rolling_ic_weights(
    factors: Dict[str, pd.DataFrame],
    forward_ret: pd.DataFrame,
    index=None,
    ic_period: int = 20,
    ic_window: int = 60,
    min_ic_obs: int = 20,
) -> pd.DataFrame:
    """按历史 IC 构造滚动权重 (index=日期, columns=因子名)。

    第 t 行的权重只用 t-ic_period 以前的 IC：
    因子在 d 日的 IC 需要 d+ic_period 的价格才能算出来，所以
    `shift(ic_period)` 之后才是 t 日真正"已知"的信息。
    """
    if ic_period <= 0:
        raise ValueError("ic_period must be positive")
    if index is None:
        index = next(iter(factors.values())).index

    cols = {}
    for name, df in factors.items():
        ic = compute_ic(df, forward_ret).ic_series.reindex(index)
        cols[name] = (
            ic.rolling(ic_window, min_periods=min_ic_obs)
              .mean()
              .shift(ic_period)
        )
    return pd.DataFrame(cols, index=index)


def ic_weighted_score(
    close: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.DataFrame] = None,
    factors: Optional[Iterable[str]] = None,
    ic_period: int = 20,
    ic_window: int = 60,
    min_ic_obs: int = 20,
    *,
    winsorize: bool = False,
    industry=None,
    log_mktcap: Optional[pd.DataFrame] = None,
    orthogonalize: bool = False,
) -> pd.DataFrame:
    """按滚动 IC 给因子加权，方向跟着历史 IC 走。

    等权合成有一个隐性假设：每个因子的正向就是"越大越好"。实际数据
    里这个假设经常不成立 —— 比如动量因子在 A 股某些年份 IC 为负，
    等权相加反而互相抵消。

    做法：
        1. 每个因子先做横截面 z-score
        2. 用 ic_period 日前向收益算逐日 IC 序列
        3. 在 t 日，只用 t-ic_period 之前的 IC 做 rolling 均值
           （t 日的 IC 要等 ic_period 天后才知道，不能拿来用）
        4. 权重 = 过去 ic_window 个 IC 的均值；负 IC 自动变成反向使用
        5. score_t = Σ w_f(t) × z_f(t) / Σ |w_f(t)|

    权重前 min_ic_obs 个点不足时该因子不参与；所有因子都没准备好时
    这一天没有分数（NaN），选股会继续等。

    v0.19 的缩尾/行业/市值/正交化参数和 composite_score 一致，并且
    IC 也在预处理后的因子上估计 —— 用原始动量算 IC、用中性化动量
    打分，两边的信号不是同一个东西。
    """
    library = make_factors(close, volume, turnover)
    names = list(factors) if factors else [n for n in DEFAULT_FACTORS if n in library]

    unknown = [n for n in names if n not in library]
    if unknown:
        raise ValueError(
            f"未知因子: {', '.join(unknown)}；可用: {', '.join(library.keys())}"
        )
    if not names:
        raise ValueError("没有可用因子")
    if ic_period <= 0:
        raise ValueError("ic_period must be positive")

    chosen = prepare_factors(
        library, names,
        winsorize=winsorize,
        industry=industry,
        log_mktcap=log_mktcap,
        orthogonalize=orthogonalize,
    )
    fwd = compute_forward_returns(close, periods=ic_period)
    weights = rolling_ic_weights(chosen, fwd, index=close.index,
                                 ic_period=ic_period,
                                 ic_window=ic_window,
                                 min_ic_obs=min_ic_obs)

    zs = {}
    valid_any = pd.DataFrame(False, index=close.index, columns=close.columns)
    for n, df in chosen.items():
        z = zscore_cross_section(df.where(df.notna()))
        zs[n] = z.where(df.notna())
        valid_any |= df.notna()

    numerator = None
    for n in names:
        w = weights[n]
        term = zs[n].multiply(w, axis=0)
        numerator = term if numerator is None else numerator.add(term, fill_value=0.0)

    denom = weights.abs().sum(axis=1)
    score = numerator.div(denom.replace(0, np.nan), axis=0)
    return score.where(valid_any & denom.notna().to_numpy()[:, None])


def select_top(
    score: pd.DataFrame,
    top_n: int = 3,
    rebalance_days: int = 5,
    min_score: Optional[float] = None,
) -> pd.DataFrame:
    """按分数选股，返回 0/1 的持仓矩阵。

    Parameters
    ----------
    score          : composite_score 的输出
    top_n          : 每次选分数最高的几只
    rebalance_days : 每多少个交易日重新排序一次
    min_score      : 分数低于它的股票不买（None 表示不设下限）

    在调仓日如果没有足够多的有效分数（比如开头的 20 天因子全是 NaN），
    就不动持仓；第一次出现足够样本后才建仓。
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if rebalance_days <= 0:
        raise ValueError("rebalance_days must be positive")

    selected = pd.DataFrame(False, index=score.index, columns=score.columns)
    current = []

    for i, date in enumerate(score.index):
        if i % rebalance_days == 0:
            row = score.loc[date].dropna()
            if min_score is not None:
                row = row[row >= min_score]
            if len(row) > 0:
                # 稳定排序：分数相同时保持列的原始顺序，结果可复现
                ranked = row.sort_values(ascending=False, kind="stable")
                current = list(ranked.index[:top_n])

        if current:
            selected.loc[date, current] = True

    return selected


def signals_from_selection(
    selected: pd.DataFrame,
    data: Dict[str, pd.DataFrame],
) -> Dict[str, pd.Series]:
    """把持仓矩阵转成组合引擎要的 per-symbol 信号。

    组合引擎的 _align() 是按"第几行"把信号塞进标的自己的日期索引的，
    所以这里必须按标的自身的日期顺序输出，而不是直接返回宽表。
    停牌/缺数据的日期用 ffill 保持上一次的持仓状态。
    """
    out: Dict[str, pd.Series] = {}

    for symbol, df in data.items():
        dates = pd.to_datetime(df["date"])
        s = selected[symbol].reindex(dates).ffill().fillna(False)
        # 引擎内部会 set_index("date").sort_index()，按排序后的日期给值
        s = pd.Series(s.to_numpy(), index=dates).sort_index()
        out[symbol] = s.astype(int)

    return out


def selection_turnover(selected: pd.DataFrame) -> dict:
    """统计选股结果的变化情况，用来看换手是不是太频繁。"""
    if selected.empty:
        return {"n_days": 0, "avg_names": 0.0, "changed_days": 0,
                "avg_change_pct": 0.0}

    active = selected.astype(bool)
    names = active.sum(axis=1)

    changed = 0
    change_pct = []
    prev = None
    for _, row in active.iterrows():
        cur = set(row.index[row.to_numpy()])
        if prev is not None and cur != prev:
            changed += 1
            if prev or cur:
                union = prev | cur
                change_pct.append(len(prev ^ cur) / len(union) if union else 0.0)
        prev = cur

    return {
        "n_days": int(len(active)),
        "avg_names": float(names[names > 0].mean()) if (names > 0).any() else 0.0,
        "changed_days": changed,
        "avg_change_pct": float(np.mean(change_pct)) if change_pct else 0.0,
    }
