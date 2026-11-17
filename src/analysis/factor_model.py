"""
因子归因 —— 组合的收益到底来自哪些风格。

前面的收益归因（attribution.py）拆到"哪只股票赚的"，Brinson 拆到
"哪个行业选得好"。因子归因再往上退一层：把组合的日收益回归到几个
**风格因子**上，回答"这段时间的收益多少来自市场、多少来自小盘、
多少来自动量"。

三因子都从回测数据里自己构造，不依赖外部因子库：

    MKT   市场因子   = 全池等权日收益
    SMB   规模因子   = 小市值组等权收益 - 大市值组等权收益
    MOM   动量因子   = 过去 N 日涨幅高的组 - 涨幅低的组

两条不写未来函数的规矩：

    1. 分组用的是 t-1 收盘时才能知道的信息（市值、过去 N 日收益），
       收益用的是 t 日的。
    2. 回归只解释同期收益，因子收益本身也不会偷看未来。

回归模型：

    r_p,t = alpha + b_mkt * MKT_t + b_smb * SMB_t + b_mom * MOM_t + eps_t

beta 是风格暴露，alpha 是扣掉风格后的超额收益，eps 是选股带来的
特异收益。收益分解把这三部分加起来，恒等于区间内的组合收益之和。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "FactorModelResult",
    "build_factor_returns",
    "estimate_exposures",
    "out_of_sample_attribution",
    "factor_contribution",
    "format_factor_report",
]

FACTOR_NAMES = ("MKT", "SMB", "MOM")


# ===================================================================
# 因子收益构造
# ===================================================================

def _group_spread(rets: pd.DataFrame, score: pd.DataFrame,
                  long_high: bool = True) -> pd.Series:
    """按横截面 score 把股票分两组，返回"高分组 - 低分组"的等权收益。

    score 是 t-1 就能知道的值（市值、过去涨幅），rets 是 t 日收益。
    NaN 值不参与分组；某一侧当天没有成员时返回 NaN。
    """
    rank = score.rank(axis=1, method="first", pct=True)
    high = rank > 0.5
    low = rank <= 0.5

    hi_ret = rets.where(high).mean(axis=1)
    lo_ret = rets.where(low).mean(axis=1)
    spread = hi_ret - lo_ret if long_high else lo_ret - hi_ret
    return spread


def build_factor_returns(close: pd.DataFrame,
                         mktcap: Optional[pd.DataFrame] = None,
                         mom_window: int = 20) -> pd.DataFrame:
    """构造 MKT / SMB / MOM 三个日频因子收益。

    Parameters
    ----------
    close     : index=日期, columns=股票代码, 收盘价（已复权）
    mktcap    : 可选, 流通市值宽表, 形状与 close 对齐。没有市值数据时
                SMB 整列为 NaN（报告里会显示"不可用"）
    mom_window: 动量因子的回看天数

    Returns
    -------
    DataFrame, index=日期, columns=MKT/SMB/MOM
    """
    if "date" in close.columns:
        px = close.set_index("date")
    else:
        px = close.copy()
    px = px.sort_index()
    rets = px.pct_change()

    # ---------- MKT: 全池等权 ----------
    mkt = rets.mean(axis=1)

    # ---------- SMB: 小市值 - 大市值 ----------
    smb = pd.Series(np.nan, index=px.index, name="SMB")
    if mktcap is not None:
        cap = mktcap.copy()
        if "date" in cap.columns:
            cap = cap.set_index("date")
        cap = cap.reindex(index=px.index, columns=px.columns).sort_index()

        # t-1 的市值决定 t 日的分组; 停牌日的空值向前补, 补不上就跳过
        cap_prev = cap.ffill().shift(1)
        smb = _group_spread(rets, cap_prev, long_high=False)
        smb.name = "SMB"

    # ---------- MOM: 过去 N 日涨幅高 - 低 ----------
    # past_t = close_{t-1} / close_{t-1-N} - 1, 不含当天
    past = px.ffill().shift(1) / px.ffill().shift(1 + mom_window) - 1.0
    mom = _group_spread(rets, past, long_high=True)
    mom.name = "MOM"

    return pd.DataFrame({"MKT": mkt, "SMB": smb, "MOM": mom})


# ===================================================================
# 回归
# ===================================================================

@dataclass
class FactorModelResult:
    """因子回归结果"""
    betas: pd.Series                # 每个因子的暴露
    tstats: pd.Series               # 显著性
    alpha_daily: float              # 日频 alpha
    alpha_tstat: float
    r2: float
    n_obs: int
    residuals: pd.Series            # 特异收益 (日频)
    factor_returns: pd.DataFrame    # 用到的因子收益
    portfolio_returns: pd.Series
    rolling_betas: Optional[pd.DataFrame] = None
    extra: dict = field(default_factory=dict)


def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    """最小二乘 + 系数 t 值。X 的第一列是常数项。"""
    n, k = X.shape
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef

    dof = n - k
    if dof > 0:
        sigma2 = float(resid @ resid) / dof
        xtx_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    else:
        se = np.full(k, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, coef / se,
                         np.where(coef != 0, np.inf, 0.0))
        tstat = np.where(np.isnan(se), np.nan, tstat)

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {"coef": coef, "tstat": tstat, "resid": resid, "r2": r2}


def _prepare(portfolio_returns: pd.Series,
             factor_returns: pd.DataFrame) -> pd.DataFrame:
    """对齐组合收益和因子收益，去掉缺数据的日期和整列缺失的因子。"""
    pr = pd.Series(portfolio_returns).copy()
    if "date" in getattr(pr, "columns", []):
        pr = pr.set_index("date")
    pr.index = pd.to_datetime(pr.index)

    fr = factor_returns.copy()
    if "date" in fr.columns:
        fr = fr.set_index("date")
    fr.index = pd.to_datetime(fr.index)

    # SMB 没有市值数据时整列是 NaN, 直接从回归里去掉
    fr = fr.dropna(axis=1, how="all")

    # 只用两边都有数的日期
    joined = pd.concat([pr.rename("ret"), fr], axis=1, join="inner")
    usable = [c for c in fr.columns if c in joined.columns]
    return joined.dropna(subset=usable + ["ret"])


def estimate_exposures(portfolio_returns: pd.Series,
                       factor_returns: pd.DataFrame,
                       window: Optional[int] = None) -> FactorModelResult:
    """把组合日收益回归到因子收益上。

    Parameters
    ----------
    portfolio_returns : index=日期 的日收益率
    factor_returns    : build_factor_returns 的输出
    window            : 可选, 给定时额外算滚动 beta（窗口长度）

    Returns
    -------
    FactorModelResult
    """
    joined = _prepare(portfolio_returns, factor_returns)
    usable = [c for c in joined.columns if c != "ret"]

    if len(joined) < len(usable) + 2:
        raise ValueError("有效样本太少，无法做因子回归")

    y = joined["ret"].to_numpy(dtype=float)
    F = joined[usable].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(y)), F])
    fit = _ols(y, X)

    betas = pd.Series(fit["coef"][1:], index=usable, name="beta")
    tstats = pd.Series(fit["tstat"][1:], index=usable, name="t")

    rolling = None
    if window and window >= len(usable) + 2 and window < len(joined):
        rows = {}
        for i in range(window, len(joined) + 1):
            sub = joined.iloc[i - window:i]
            Xw = np.column_stack([np.ones(len(sub)),
                                  sub[usable].to_numpy(dtype=float)])
            cw, *_ = np.linalg.lstsq(Xw, sub["ret"].to_numpy(dtype=float),
                                     rcond=None)
            rows[sub.index[-1]] = cw[1:]
        rolling = pd.DataFrame.from_dict(rows, orient="index",
                                         columns=usable).sort_index()

    return FactorModelResult(
        betas=betas,
        tstats=tstats,
        alpha_daily=float(fit["coef"][0]),
        alpha_tstat=float(fit["tstat"][0]),
        r2=float(fit["r2"]),
        n_obs=len(joined),
        residuals=pd.Series(fit["resid"], index=joined.index, name="residual"),
        factor_returns=joined[usable],
        portfolio_returns=joined["ret"],
        rolling_betas=rolling,
    )


def out_of_sample_attribution(portfolio_returns: pd.Series,
                              factor_returns: pd.DataFrame,
                              split: float = 0.5):
    """样本外收益分解：前一半估 beta，后一半做分解。

    样本内回归带截距，残差和恒等于 0（残差与常数项正交），所以
    "特异收益"那一栏永远是 0，什么也说明不了。把 beta 固定在前半段，
    再拿去解释后半段，残差就变成真正的样本外选股贡献了。

    Returns
    -------
    (model, contribution) —— model 是前半段估出来的暴露;
                            contribution 的 attrs 里带 split_date
    """
    joined = _prepare(portfolio_returns, factor_returns)
    usable = [c for c in joined.columns if c != "ret"]
    if not 0.0 < split < 1.0:
        raise ValueError("split 必须在 (0, 1) 之间")

    cut = int(len(joined) * split)
    if cut < len(usable) + 2 or len(joined) - cut < 2:
        raise ValueError("样本太少，无法做样本外分解")

    est = joined.iloc[:cut]
    test = joined.iloc[cut:]
    model = estimate_exposures(est["ret"], est[usable])

    pred = model.alpha_daily * 1.0
    for name in model.betas.index:
        pred = pred + model.betas[name] * test[name]
    resid = test["ret"] - pred

    rows = [{"item": "alpha", "beta": np.nan,
             "contribution": model.alpha_daily * len(test)}]
    for name in model.betas.index:
        rows.append({"item": name, "beta": float(model.betas[name]),
                     "contribution": float((model.betas[name] * test[name]).sum())})
    rows.append({"item": "特异收益", "beta": np.nan,
                 "contribution": float(resid.sum())})

    out = pd.DataFrame(rows)
    total = float(test["ret"].sum())
    out["share"] = np.where(total != 0, out["contribution"] / total, np.nan)
    out.attrs["total_return"] = total
    out.attrs["identity_error"] = total - out["contribution"].sum()
    out.attrs["mode"] = "oos"
    out.attrs["split_date"] = str(test.index[0].date())
    out.attrs["n_est"] = len(est)
    out.attrs["n_test"] = len(test)
    return model, out


# ===================================================================
# 收益分解
# ===================================================================

def factor_contribution(model: FactorModelResult) -> pd.DataFrame:
    """把区间累计收益拆成 alpha / 各因子 / 特异收益三部分。

    日频收益直接算术相加（不是复利），这样三部分之和严格等于组合
    日收益之和。报告里会注明这是算术口径。
    """
    total_ret = float(model.portfolio_returns.sum())
    rows = []

    alpha_total = model.alpha_daily * model.n_obs
    rows.append({"item": "alpha", "beta": np.nan,
                 "contribution": alpha_total})

    for name in model.betas.index:
        contrib = float((model.betas[name] * model.factor_returns[name]).sum())
        rows.append({"item": name, "beta": float(model.betas[name]),
                     "contribution": contrib})

    resid = float(model.residuals.sum())
    rows.append({"item": "特异收益", "beta": np.nan, "contribution": resid})

    out = pd.DataFrame(rows)
    out["share"] = np.where(total_ret != 0,
                            out["contribution"] / total_ret, np.nan)
    out.attrs["total_return"] = total_ret
    out.attrs["identity_error"] = total_ret - out["contribution"].sum()
    out.attrs["mode"] = "in_sample"
    return out


def format_factor_report(model: FactorModelResult,
                         contribution: Optional[pd.DataFrame] = None) -> str:
    """终端文本报告。"""
    lines = []
    lines.append("-" * 62)
    lines.append("  因子归因（自建 MKT / SMB / MOM 三因子）")
    lines.append("-" * 62)
    lines.append(f"  样本: {model.n_obs} 个交易日    R^2 = {model.r2:.3f}")
    lines.append(f"  日频 alpha = {model.alpha_daily * 10000:+.2f} bp   "
                 f"t = {model.alpha_tstat:+.2f}")
    lines.append("")
    lines.append("  暴露")
    lines.append("  " + "-" * 58)
    lines.append(f"  {'因子':<6}{'beta':>10}{'t 值':>10}   说明")
    meaning = {
        "MKT": "对全池等权收益的敏感度",
        "SMB": "正 = 偏小盘, 负 = 偏大盘",
        "MOM": "正 = 偏近期强势股",
    }
    for name in model.betas.index:
        lines.append(f"  {name:<6}{model.betas[name]:>10.3f}"
                     f"{model.tstats[name]:>10.2f}   {meaning.get(name, '')}")

    contrib = contribution if contribution is not None else factor_contribution(model)
    total = contrib.attrs.get("total_return", np.nan)
    lines.append("")
    lines.append("  收益分解（算术口径，区间累计）")
    lines.append("  " + "-" * 58)
    lines.append(f"  {'来源':<8}{'贡献':>12}{'占比':>10}")
    for _, row in contrib.iterrows():
        share = "" if np.isnan(row["share"]) else f"{row['share'] * 100:>9.1f}%"
        lines.append(f"  {row['item']:<8}{row['contribution'] * 100:>+11.2f}%"
                     f"{share:>10}")
    lines.append(f"  {'合计':<8}{total * 100:>+11.2f}%")
    err = contrib.attrs.get("identity_error", 0.0)
    lines.append(f"  恒等式误差: {err:.2e}")

    if contrib.attrs.get("mode") == "oos":
        lines.append("")
        lines.append(f"  样本外分解: 前 {contrib.attrs['n_est']} 天估 beta, "
                     f"从 {contrib.attrs['split_date']} 起共 "
                     f"{contrib.attrs['n_test']} 天用固定 beta 解释")
    else:
        lines.append("")
        lines.append("  注: 样本内回归的残差总和恒为 0（带截距 OLS 的性质），")
        lines.append("      '特异收益'只用来校验恒等式。要看选股贡献请用样本外分解。")

    if model.rolling_betas is not None and len(model.rolling_betas):
        last = model.rolling_betas.iloc[-1]
        lines.append("")
        lines.append("  最近一期滚动暴露")
        lines.append("  " + "-" * 58)
        lines.append("  " + "  ".join(f"{k}={v:.3f}" for k, v in last.items()))

    return "\n".join(lines)
