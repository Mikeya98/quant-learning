"""
收益归因 —— 回测赚的钱到底是谁赚的。

绩效指标回答"赚了多少"，归因回答"从哪赚的"：

    个券贡献   把组合每日收益拆到每只股票：前一天的权重 × 当天涨幅。
               权重用 t-1 收盘的，收益用 t 日的，避免拿当天收盘后才知道的
               权重去解释当天收益。
    残差       实际净值收益 - 各标的贡献之和。收盘成交模式下它主要就是
               手续费和印花税；次日开盘模式下还包含"开盘买入、当天涨跌"
               这部分时点收益。
    Brinson    相对基准的超额收益拆成三块：
                  配置效应  (组合行业权重 - 基准行业权重) × 基准行业收益
                  选择效应  基准行业权重 × (组合行业收益 - 基准行业收益)
                  交互效应  两者都偏离带来的交叉项
               三项相加等于组合收益减基准收益，这个恒等式是检查代码
               有没有写错的最好工具。

基准默认是"等权买入持有"：期初每只股票买相同的金额，之后不再调仓，
权重随涨跌自然漂移。组合有现金时，把现金当成一个收益为 0 的行业，
这样 Brinson 的恒等式仍然成立。
"""

import numpy as np
import pandas as pd

_DAILY_COLUMNS = ("market", "equity_ret", "residual")


def _date_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(df["date"]))


def _weights_panel(weights_df: pd.DataFrame, symbols) -> pd.DataFrame:
    """把 weights_df (date + 每列一个标的) 转成日期索引、只保留给定标的。"""
    if "date" not in weights_df.columns:
        raise ValueError("weights_df 缺少 date 列")
    w = weights_df.copy()
    w["date"] = pd.to_datetime(w["date"])
    w = w.set_index("date").sort_index()
    return w.reindex(columns=list(symbols)).fillna(0.0)


def daily_contributions(prices: pd.DataFrame, weights_df: pd.DataFrame,
                        equity_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    每日收益归因。

    Parameters
    ----------
    prices      index=日期, columns=标的, 收盘价
    weights_df  组合回测输出的权重表 (date + 标的列)
    equity_df   可选, 含 date / equity 两列。给了就计算残差。

    Returns
    -------
    DataFrame, index=日期, columns=每个标的 + market / equity_ret / residual
    """
    symbols = list(prices.columns)
    px = prices.sort_index().ffill()
    rets = px.pct_change().fillna(0.0)

    w = _weights_panel(weights_df, symbols).reindex(px.index).fillna(0.0)
    w_prev = w.shift(1).fillna(0.0)

    contrib = w_prev * rets[symbols]
    out = contrib.copy()
    out["market"] = contrib.sum(axis=1)

    if equity_df is not None:
        eq = equity_df.copy()
        eq["date"] = pd.to_datetime(eq["date"])
        eq = eq.set_index("date")["equity"].sort_index().reindex(px.index)
        eq_ret = eq.pct_change().fillna(0.0)
        out["equity_ret"] = eq_ret
        out["residual"] = eq_ret - out["market"]

    return out


def contribution_summary(daily: pd.DataFrame,
                         weights_df: pd.DataFrame) -> pd.DataFrame:
    """
    把每日贡献按标的汇总。

    返回列：
        symbol, total, avg_daily, active_days, win_rate_pct, share_pct
    total 是每日收益率的算术和（不是复利），share_pct 是它占全部标的
    贡献之和的比例。
    """
    symbols = [c for c in weights_df.columns if c != "date"]
    symbols = [s for s in symbols if s in daily.columns]

    w = _weights_panel(weights_df, symbols)
    w_prev = w.reindex(daily.index).shift(1).fillna(0.0)
    active = w_prev > 1e-12

    rows = []
    for s in symbols:
        c = daily[s]
        act = active[s]
        active_days = int(act.sum())
        pos_days = int(((c > 0) & act).sum())
        rows.append({
            "symbol": s,
            "total": float(c.sum()),
            "avg_daily": float(c.mean()),
            "active_days": active_days,
            "win_rate_pct": (pos_days / active_days * 100) if active_days else 0.0,
        })

    df = pd.DataFrame(rows)
    total_all = df["total"].sum()
    df["share_pct"] = (df["total"] / total_all * 100) if total_all != 0 else 0.0
    df["share_pct"] = df["share_pct"].fillna(0.0)
    return df.sort_values("total", ascending=False).reset_index(drop=True)


def benchmark_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """等权买入持有的每日权重：期初等金额，之后随价格漂移。"""
    px = prices.sort_index().ffill()
    base = px.iloc[0].replace(0, np.nan)
    value = px.divide(base, axis=1)
    total = value.sum(axis=1).replace(0, np.nan)
    return value.divide(total, axis=0).fillna(0.0)


def brinson_attribution(prices: pd.DataFrame,
                        weights_df: pd.DataFrame,
                        industry_map: pd.Series = None,
                        benchmark: pd.DataFrame = None) -> pd.DataFrame:
    """
    Brinson 归因（逐日计算后累加）。

    Parameters
    ----------
    prices        收盘价宽表
    weights_df    组合权重表
    industry_map  index=标的, 值=行业。为空时全部算作"全市场"。
    benchmark     基准每日权重，默认等权买入持有

    Returns
    -------
    DataFrame, index=行业, columns=allocation / selection / interaction / total
    另外在 df.attrs 里给出 portfolio_return / benchmark_return / excess /
    identity_error（恒等式两边之差，正常应接近 0）。
    """
    symbols = list(prices.columns)
    px = prices.sort_index().ffill()
    rets = px.pct_change().fillna(0.0)

    w = _weights_panel(weights_df, symbols).reindex(px.index).fillna(0.0)
    w_prev = w.shift(1).fillna(0.0)

    if benchmark is None:
        bw = benchmark_weights(px)
    else:
        bw = benchmark.reindex(px.index).reindex(columns=symbols).fillna(0.0)
    b_prev = bw.shift(1).fillna(0.0)

    if industry_map is None or len(industry_map) == 0:
        ind = pd.Series("全市场", index=symbols)
    else:
        def _industry(s):
            v = industry_map.get(s)
            return v if isinstance(v, str) and v else "未分类"
        ind = pd.Series({s: _industry(s) for s in symbols})

    industries = list(pd.unique(ind.values))
    alloc = {i: 0.0 for i in industries}
    select = {i: 0.0 for i in industries}
    inter = {i: 0.0 for i in industries}

    for _, s in ind.groupby(ind).groups.items():
        group = list(s)
        wp = w_prev[group].sum(axis=1)
        wb = b_prev[group].sum(axis=1)
        # 行业收益按权重加权
        rp = (w_prev[group] * rets[group]).sum(axis=1) / wp.replace(0, np.nan)
        rb = (b_prev[group] * rets[group]).sum(axis=1) / wb.replace(0, np.nan)
        rp = rp.fillna(0.0)
        rb = rb.fillna(0.0)

        # 只有一边持有这个行业时, 用另一边的行业收益补齐。
        # 否则"组合没买、基准涨了"会被拆成相反的选择/交互效应,
        # 数字对但读起来像是选股亏损。补齐后恒等式仍然成立。
        rp = rp.where(wp > 0, rb)
        rb = rb.where(wb > 0, rp)

        name = ind[group[0]]
        alloc[name] = float(((wp - wb) * rb).sum())
        select[name] = float((wb * (rp - rb)).sum())
        inter[name] = float(((wp - wb) * (rp - rb)).sum())

    df = pd.DataFrame({
        "allocation": pd.Series(alloc),
        "selection": pd.Series(select),
        "interaction": pd.Series(inter),
    })
    df["total"] = df.sum(axis=1)
    df = df.sort_values("total", ascending=False)

    # 恒等式检查
    port_ret = float((w_prev * rets[symbols]).sum().sum())
    bench_ret = float((b_prev * rets[symbols]).sum().sum())
    excess = port_ret - bench_ret
    identity_error = float(df["total"].sum()) - excess

    df.attrs["portfolio_return"] = port_ret
    df.attrs["benchmark_return"] = bench_ret
    df.attrs["excess"] = excess
    df.attrs["identity_error"] = identity_error
    return df


def format_attribution_report(summary: pd.DataFrame,
                              brinson: pd.DataFrame) -> str:
    """终端输出：个券贡献 + Brinson 三效应。"""
    lines = []
    lines.append("=" * 62)
    lines.append("  收益归因")
    lines.append("=" * 62)

    lines.append("")
    lines.append("-" * 62)
    lines.append("  个券贡献（每日 前日权重 × 当日涨幅 的累加）")
    lines.append("-" * 62)
    lines.append(f"  {'标的':<10}{'累计贡献':>12}{'占比':>10}"
                 f"{'持仓天数':>10}{'胜率':>10}")
    for _, r in summary.iterrows():
        lines.append(
            f"  {r['symbol']:<10}{r['total'] * 100:>11.2f}%"
            f"{r['share_pct']:>9.1f}%{int(r['active_days']):>10d}"
            f"{r['win_rate_pct']:>9.1f}%"
        )

    lines.append("")
    lines.append("-" * 62)
    lines.append("  Brinson 归因（相对等权买入持有基准）")
    lines.append("-" * 62)
    lines.append(f"  {'行业':<12}{'配置':>12}{'选择':>12}"
                 f"{'交互':>12}{'合计':>12}")
    for name, r in brinson.iterrows():
        lines.append(
            f"  {name:<12}{r['allocation'] * 100:>11.2f}%"
            f"{r['selection'] * 100:>11.2f}%"
            f"{r['interaction'] * 100:>11.2f}%"
            f"{r['total'] * 100:>11.2f}%"
        )

    a = brinson.attrs
    lines.append("")
    lines.append(f"  组合累计收益（日收益算术和）: {a['portfolio_return'] * 100:+.2f}%")
    lines.append(f"  基准累计收益（日收益算术和）: {a['benchmark_return'] * 100:+.2f}%")
    lines.append(f"  超额收益:                     {a['excess'] * 100:+.2f}%")
    lines.append(f"  恒等式误差:                   {a['identity_error'] * 100:+.6f}%")
    lines.append("")
    lines.append("  注：逐日效应直接累加，和复利口径的净值收益略有差异。")
    return "\n".join(lines)
