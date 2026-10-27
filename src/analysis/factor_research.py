"""
因子研究 —— IC 分析、分层回测、多因子合成。

前面几个阶段都是"给定策略参数，看回测结果"。到了因子这一步，问题
反过来了：**先判断一个指标对未来收益有没有预测能力**，再决定要不要
拿它去做策略。

核心工具是 IC（Information Coefficient）：

    IC_t = corr( 因子值_t , 未来 N 日收益_t )   —— 每一天做一次横截面相关

把所有交易日的 IC 连起来，看它的均值、稳定性（ICIR）、胜率。
一个日频因子 IC 均值能稳定在 0.03~0.05 就算不错了，0.1 以上通常是
数据有问题（未来函数）。

本模块只依赖 pandas / numpy，所有函数按"宽表"约定：
    index   = 交易日
    columns = 股票代码

约定:
    - 因子值越大越看好的因子，IC 应该为正；反向因子在构造时取负号
    - 分层回测里 Q1 是因子值最小的一组，Qn 是最大的一组
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

__all__ = [
    "ICResult",
    "compute_forward_returns",
    "compute_ic",
    "quantile_returns",
    "quantile_summary",
    "zscore_cross_section",
    "combine_factors",
    "make_factors",
    "format_ic_report",
]


# ===================================================================
# 未来收益
# ===================================================================

def compute_forward_returns(close: pd.DataFrame, periods: int = 5) -> pd.DataFrame:
    """
    计算未来 periods 个交易日的收益。

        fwd_ret_t = close_{t+periods} / close_t - 1

    最后 periods 行为 NaN —— 这部分没有未来数据，不能参与 IC 计算。
    回测里最容易犯的错就是这里忘了 shift，结果因子和收益在同一天，
    IC 直接虚高到 0.9。

    Parameters
    ----------
    close   : 宽表，index=交易日，columns=股票代码，值=收盘价
    periods : 持有期（交易日）

    Returns
    -------
    同形状的 DataFrame，前向收益
    """
    if periods <= 0:
        raise ValueError("periods must be positive")

    fwd = close.shift(-periods) / close - 1.0
    return fwd


# ===================================================================
# IC 分析
# ===================================================================

@dataclass
class ICResult:
    """一段区间内 IC 序列的统计结果"""

    method: str
    periods: int
    ic_series: pd.Series          # index=交易日
    mean_ic: float
    std_ic: float
    icir: float                   # mean / std（不年化）
    t_stat: float                 # mean / (std / sqrt(n))
    positive_ratio: float         # IC > 0 的比例
    n_samples: int                # 有效交易日数

    def __repr__(self) -> str:  # 方便在 notebook 里直接看
        return (f"ICResult({self.method}, period={self.periods}d, "
                f"mean={self.mean_ic:.4f}, ICIR={self.icir:.3f}, "
                f"t={self.t_stat:.2f}, n={self.n_samples})")


def compute_ic(
    factor: pd.DataFrame,
    forward_ret: pd.DataFrame,
    method: str = "spearman",
    min_obs: int = 5,
) -> ICResult:
    """
    逐日计算横截面 IC。

    Parameters
    ----------
    factor      : 因子宽表（已对齐到和 forward_ret 相同的 index/columns）
    forward_ret : 未来收益宽表
    method      : "pearson" | "spearman"（默认，即 RankIC，对异常值不敏感）
    min_obs     : 当天至少有多少只股票才计算（样本太少相关没有意义）

    Returns
    -------
    ICResult
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")

    factor, forward_ret = factor.align(forward_ret, join="inner")

    # 股票数不够时不报错, 返回空结果 —— 调用方可以把它当成
    # "这个区间算不出 IC" 正常处理, 不用每个调用点都包 try。
    if factor.shape[1] < min_obs:
        return ICResult(method, 0, pd.Series(dtype=float),
                        0.0, 0.0, 0.0, 0.0, 0.0, 0)

    ic_values = {}
    for dt in factor.index:
        f = factor.loc[dt]
        r = forward_ret.loc[dt]
        mask = f.notna() & r.notna()
        if mask.sum() < min_obs:
            continue

        fv = f[mask].astype(float)
        rv = r[mask].astype(float)

        # 常数因子（例如全市场停牌）算不出相关，跳过
        if fv.nunique() < 2 or rv.nunique() < 2:
            continue

        corr = fv.corr(rv, method=method)
        if pd.notna(corr):
            ic_values[dt] = corr

    ic_series = pd.Series(ic_values, dtype=float).sort_index()
    n = len(ic_series)

    if n == 0:
        return ICResult(method, 0, ic_series, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std(ddof=1)) if n > 1 else 0.0
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0.0
    positive_ratio = float((ic_series > 0).mean())

    periods = 0  # 由调用方按需填写
    return ICResult(method, periods, ic_series, mean_ic, std_ic,
                    icir, t_stat, positive_ratio, n)


# ===================================================================
# 分层（quantile）回测
# ===================================================================

def quantile_returns(
    factor: pd.DataFrame,
    forward_ret: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    按因子值把股票分成 n 层，计算每层未来的平均收益。

    返回的 DataFrame:
        index   = 交易日
        columns = Q1 ... Qn（Q1 = 因子值最小的那一组）

    如果因子真的有效，这个表应该呈现单调性：
    多因子 Q1 < Q2 < ... < Qn，或者反过来（因子取负号之后应该统一）。

    注意：这里只是"分组看均值"，没有考虑交易成本、调仓频率、
    涨跌停不可成交。它是判断因子有没有信息的第一步，不是策略净值。
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2")

    factor, forward_ret = factor.align(forward_ret, join="inner")
    cols = [f"Q{i + 1}" for i in range(n_quantiles)]
    rows = {}

    for dt in factor.index:
        f = factor.loc[dt]
        r = forward_ret.loc[dt]
        mask = f.notna() & r.notna()
        if mask.sum() < n_quantiles:
            continue

        fv = f[mask].astype(float)
        rv = r[mask].astype(float)

        # rank(method="first") 保证因子值相同的股票也能均分，
        # 用 qcut 遇到大量重复值会直接抛异常。
        ranks = fv.rank(method="first")
        labels = pd.qcut(ranks, n_quantiles, labels=False)
        rows[dt] = rv.groupby(labels).mean()

    if not rows:
        return pd.DataFrame(columns=cols)

    qt = pd.DataFrame(rows).T.sort_index()
    qt.columns = [cols[int(c)] for c in qt.columns]
    return qt.reindex(columns=cols)


def quantile_summary(qret: pd.DataFrame, periods_per_year: int = 252,
                     holding_period: int = 5) -> pd.DataFrame:
    """
    把分层收益汇总成"每层平均收益 / 年化 / 胜率"。

    这里的"年化"是把每期收益按持有期折算，属于粗略估计，
    不同调仓频率下不能直接比较。
    """
    if qret.empty:
        return pd.DataFrame()

    scale = periods_per_year / holding_period
    summary = pd.DataFrame({
        "mean": qret.mean(),
        "std": qret.std(),
        "annual": (1 + qret.mean()) ** scale - 1,
        "win_rate": (qret > 0).mean(),
        "n": qret.notna().sum(),
    })
    summary["sharpe"] = summary["mean"] / summary["std"]
    return summary


# ===================================================================
# 多因子合成
# ===================================================================

def zscore_cross_section(df: pd.DataFrame) -> pd.DataFrame:
    """
    横截面标准化：每个交易日单独做 (x - mean) / std。

    不同因子的量纲差很多（动量是百分比，换手率是倍数），
    合成之前必须先标准化。这里按行做，因为因子之间是可比的
    只有同一天同一批股票。
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1, ddof=0)
    # 某天只有一只股票、或者全部相同 → std=0，直接给 0 分
    std = std.replace(0, np.nan)
    z = df.sub(mean, axis=0).div(std, axis=0)
    return z.fillna(0.0)


def combine_factors(
    factors: Dict[str, pd.DataFrame],
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    多因子等权/加权合成。

    每个因子先做横截面 z-score，再按权重相加。
    权重为 None 时等权。

    真实的多因子模型会按 IC 加权、做因子正交化、控制行业暴露，
    这里保留最简单的版本，至少把"多因子打分"这件事跑通。
    """
    if not factors:
        raise ValueError("factors is empty")

    if weights is None:
        weights = {k: 1.0 for k in factors}

    total = 0.0
    score = None
    for name, df in factors.items():
        w = float(weights.get(name, 0.0))
        if w == 0.0:
            continue
        z = zscore_cross_section(df)
        score = z * w if score is None else score.add(z * w, fill_value=np.nan)
        total += abs(w)

    if score is None:
        raise ValueError("all factor weights are zero")

    return score / total if total else score


# ===================================================================
# 内置因子库
# ===================================================================

def make_factors(
    close: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    turnover: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    用日线数据构造几个最基础的因子。

    momentum_20  20 日动量：过去涨得多的未来还涨（趋势）
    reversal_5    5 日反转：过去涨得多的接下来跌（A 股短期反转明显，所以取负）
    vol_20       20 日波动率取负：低波动股票风险调整后更好（低波异象）
    vol_ratio    成交量 / 20 日均量：放量（资金关注度）

    这些因子都很粗糙，目的是演示研究流程，不是可直接上实盘的信号。
    """
    factors = {
        "momentum_20": close.pct_change(20),
        "reversal_5": -close.pct_change(5),
        "vol_20": -close.pct_change().rolling(20).std(),
    }

    if volume is not None:
        factors["vol_ratio"] = volume / volume.rolling(20).mean()

    if turnover is not None:
        factors["turnover_20"] = turnover.rolling(20).mean()

    return factors


# ===================================================================
# 文本报告
# ===================================================================

def format_ic_report(
    results: Dict[str, ICResult],
    qsummary: Optional[Dict[str, pd.DataFrame]] = None,
) -> str:
    """把多个因子的 IC 结果拼成一段可读的文本"""
    lines = []
    lines.append("=" * 62)
    lines.append("  因子 IC 分析")
    lines.append("=" * 62)
    lines.append(f"{'因子':<16}{'mean IC':>9}{'ICIR':>8}{'t值':>8}{'胜率':>8}{'样本':>7}")
    lines.append("-" * 62)

    for name, r in results.items():
        lines.append(
            f"{name:<16}{r.mean_ic:>9.4f}{r.icir:>8.3f}"
            f"{r.t_stat:>8.2f}{r.positive_ratio:>8.2%}{r.n_samples:>7}")

    lines.append("-" * 62)
    lines.append("  经验参考: |mean IC| > 0.03 且 ICIR > 0.3 才值得继续研究")

    if qsummary:
        for name, qs in qsummary.items():
            if qs.empty:
                continue
            lines.append("")
            lines.append(f"  {name} 分层（持有期内平均收益）")
            header = "  " + "".join(f"{idx:>12}" for idx in qs.index)
            lines.append(header)
            lines.append("  " + "".join(f"{v:>12.4%}" for v in qs["mean"]))
            # 单调性: 相邻层差分同号的比例
            diffs = qs["mean"].diff().dropna()
            if len(diffs) > 0:
                mono = (np.sign(diffs) == np.sign(diffs.iloc[0])).mean()
                lines.append(f"  单调性: {mono:.0%}（相邻层收益方向一致的比例）")

    return "\n".join(lines)
