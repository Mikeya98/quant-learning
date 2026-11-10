"""
交易成本拆分（Phase 15）。

回测里说"成本吃了 30 个点"，往往只有一个总数，看不出钱花在哪里。
这一层把每笔成交的成本拆成两类四款：

显性成本（直接从现金里扣）:
    佣金      commission      万 2.5，最低 5 元/笔，买卖双向
    印花税    stamp_duty      千 1，仅卖出
    过户费    transfer_fee    万 0.1，买卖双向

隐性成本（不额外扣款，但体现在成交价里）:
    滑点      slippage        成交价相对决策价的价差 × 股数

注意：显性成本会真的减少现金；滑点只是"如果按决策价成交能多赚
多少"，不能和显性成本加在一起当现金支出。报告里两者分开列，
"合计"仅用于估算总拖累。

单位：bps = 成本 / 成交额 × 10000。
"""

import pandas as pd

COST_COLUMNS = ("commission", "stamp_duty", "transfer_fee", "slippage")


def normalize_trades(trades) -> pd.DataFrame:
    """补齐成本列，算出显性成本和合计成本。

    兼容旧版 trades 表（只有 fee 一列）：fee 当作显性成本，其余
    分项按 0 处理，不会因为缺列直接 KeyError。
    """
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=[
            "date", "symbol", "side", "shares", "price", "ref_price",
            "notional", "commission", "stamp_duty", "transfer_fee",
            "slippage", "fee", "explicit_cost", "total_cost",
        ])

    t = trades.copy()

    has_split = any(col in t.columns for col in COST_COLUMNS)

    if "notional" not in t.columns:
        t["notional"] = t["price"] * t["shares"]
    for col in COST_COLUMNS:
        if col not in t.columns:
            t[col] = 0.0

    # 旧版 trades 只有一个 fee 汇总列: 把它整体归到佣金, 保证总额不丢
    if not has_split and "fee" in t.columns:
        t["commission"] = t["fee"]
    if "fee" not in t.columns:
        t["fee"] = (t["commission"] + t["stamp_duty"] + t["transfer_fee"])

    t["explicit_cost"] = t["commission"] + t["stamp_duty"] + t["transfer_fee"]
    t["total_cost"] = t["explicit_cost"] + t["slippage"]
    return t


def _bps(cost: float, notional: float) -> float:
    return float(cost) / float(notional) * 10000.0 if notional > 0 else 0.0


def cost_breakdown(trades) -> dict:
    """按费用类别、买卖方向、标的三层汇总交易成本。

    Returns:
        dict: 总量指标 + by_side / by_symbol 两张表。

        总量指标里：
            explicit_cost  佣金 + 印花税 + 过户费
            slippage       滑点（隐性，不扣现金）
            total_cost     两者之和（估算的"总拖累"）
            explicit_bps   显性成本 / 成交额（万分之）
            slippage_bps   滑点 / 成交额
    """
    t = normalize_trades(trades)
    notional = float(t["notional"].sum()) if len(t) else 0.0
    commission = float(t["commission"].sum()) if len(t) else 0.0
    stamp_duty = float(t["stamp_duty"].sum()) if len(t) else 0.0
    transfer_fee = float(t["transfer_fee"].sum()) if len(t) else 0.0
    slippage = float(t["slippage"].sum()) if len(t) else 0.0
    explicit = commission + stamp_duty + transfer_fee

    if len(t):
        by_side = (t.groupby("side")[["notional", *COST_COLUMNS]]
                     .sum()
                     .reindex(["BUY", "SELL"])
                     .dropna(how="all"))
        by_side["explicit_cost"] = (by_side["commission"]
                                    + by_side["stamp_duty"]
                                    + by_side["transfer_fee"])
        by_side["total_cost"] = by_side["explicit_cost"] + by_side["slippage"]
        by_side["cost_bps"] = [
            _bps(c, n) for c, n in zip(by_side["total_cost"],
                                       by_side["notional"])
        ]

        by_symbol = (t.groupby("symbol")[["notional", *COST_COLUMNS]]
                       .sum())
        by_symbol["explicit_cost"] = (by_symbol["commission"]
                                      + by_symbol["stamp_duty"]
                                      + by_symbol["transfer_fee"])
        by_symbol["total_cost"] = (by_symbol["explicit_cost"]
                                   + by_symbol["slippage"])
        by_symbol["cost_bps"] = [
            _bps(c, n) for c, n in zip(by_symbol["total_cost"],
                                       by_symbol["notional"])
        ]
        by_symbol = by_symbol.sort_values("total_cost", ascending=False)
    else:
        by_side = pd.DataFrame()
        by_symbol = pd.DataFrame()

    return {
        "n_trades": int(len(t)),
        "notional": notional,
        "commission": commission,
        "stamp_duty": stamp_duty,
        "transfer_fee": transfer_fee,
        "explicit_cost": explicit,
        "slippage": slippage,
        "total_cost": explicit + slippage,
        "explicit_bps": _bps(explicit, notional),
        "slippage_bps": _bps(slippage, notional),
        "total_bps": _bps(explicit + slippage, notional),
        "by_side": by_side,
        "by_symbol": by_symbol,
    }


def format_cost_report(bd: dict, top_n: int = 5) -> str:
    """把 cost_breakdown 的结果排成终端报告。"""
    lines = []
    lines.append("-" * 62)
    lines.append("  交易成本拆分")
    lines.append("-" * 62)

    if bd["n_trades"] == 0:
        lines.append("  本期没有成交。")
        return "\n".join(lines)

    def pct(part):
        return part / bd["total_cost"] * 100 if bd["total_cost"] > 0 else 0.0

    lines.append(f"  成交笔数:   {bd['n_trades']:>12d}")
    lines.append(f"  成交额:     {bd['notional']:>12,.2f}")
    lines.append("")
    lines.append("  显性成本（直接扣现金）")
    lines.append(f"    佣金:     {bd['commission']:>12,.2f}"
                 f"  ({pct(bd['commission']):>5.1f}%)")
    lines.append(f"    印花税:   {bd['stamp_duty']:>12,.2f}"
                 f"  ({pct(bd['stamp_duty']):>5.1f}%)")
    lines.append(f"    过户费:   {bd['transfer_fee']:>12,.2f}"
                 f"  ({pct(bd['transfer_fee']):>5.1f}%)")
    lines.append(f"    小计:     {bd['explicit_cost']:>12,.2f}"
                 f"  ({bd['explicit_bps']:.1f} bps / 成交额)")
    lines.append("")
    lines.append("  隐性成本（已含在成交价里）")
    lines.append(f"    滑点:     {bd['slippage']:>12,.2f}"
                 f"  ({bd['slippage_bps']:.1f} bps / 成交额)")
    lines.append("")
    lines.append(f"  估算总拖累: {bd['total_cost']:>12,.2f}"
                 f"  ({bd['total_bps']:.1f} bps / 成交额)")
    lines.append("")

    by_side = bd["by_side"]
    if len(by_side):
        lines.append("  按方向")
        for side, row in by_side.iterrows():
            lines.append(
                f"    {side:<4}: 成交额 {row['notional']:>12,.0f}  "
                f"显性 {row['explicit_cost']:>9,.2f}  "
                f"滑点 {row['slippage']:>9,.2f}  "
                f"合计 {row['cost_bps']:>5.1f} bps"
            )
        lines.append("")

    by_symbol = bd["by_symbol"]
    if len(by_symbol):
        lines.append(f"  按标的（成本最高的前 {min(top_n, len(by_symbol))} 只）")
        for symbol, row in by_symbol.head(top_n).iterrows():
            lines.append(
                f"    {symbol:<10} 显性 {row['explicit_cost']:>9,.2f}  "
                f"滑点 {row['slippage']:>9,.2f}  "
                f"合计 {row['cost_bps']:>6.1f} bps"
            )

    return "\n".join(lines)
