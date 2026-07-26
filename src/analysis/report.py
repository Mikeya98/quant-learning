"""
报告生成 —— 将回测结果输出为 Markdown 格式的完整报告。

包含：
- 策略概况
- 绩效指标表
- 交易记录摘要
- 多策略对比表
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "reports"


def generate_report(results: list, save: bool = True) -> str:
    """
    生成多策略回测对比报告。

    Args:
        results: BacktestResult 对象列表
        save:    是否保存为文件

    Returns:
        Markdown 格式的报告文本
    """
    from .metrics import compute_all

    today = datetime.now().strftime("%Y-%m-%d")
    lines = []
    lines.append(f"# 量化回测报告")
    lines.append(f"")
    lines.append(f"**生成日期**: {today}")
    lines.append(f"")

    # 每个策略的详细结果
    for i, result in enumerate(results):
        m = compute_all(result)
        lines.append(f"## {i+1}. {result.strategy_name}")
        lines.append(f"")
        lines.append(f"**标的**: {result.symbol}")
        lines.append(f"")
        lines.append(f"| 指标 | 数值 |")
        lines.append(f"|------|------|")
        lines.append(f"| 总收益率 | {m['total_return_pct']:+.2f}% |")
        lines.append(f"| 年化收益率 | {m['annual_return_pct']:+.2f}% |")
        lines.append(f"| 年化波动率 | {m['annual_volatility_pct']:.2f}% |")
        lines.append(f"| 夏普比率 | {m['sharpe_ratio']:.2f} |")
        lines.append(f"| 最大回撤 | {m['max_drawdown_pct']:+.1f}% |")
        lines.append(f"| 回撤持续 | {m['max_dd_duration_days']} 天 |")
        lines.append(f"| Calmar 比率 | {m['calmar_ratio']:.2f} |")
        lines.append(f"| 交易次数 | {m['n_trades']} |")
        lines.append(f"| 胜率 | {m['win_rate_pct']:.1f}% |")
        lines.append(f"| 盈亏比 | {m['profit_factor']:.2f} |")
        lines.append(f"| 平均盈亏 | {m['avg_trade_pnl_pct']:+.2f}% |")
        lines.append(f"")

        # 最近几笔交易
        trades = result.trades_df
        if len(trades) > 0:
            lines.append(f"### 最近交易")
            lines.append(f"")
            lines.append(f"| 买入日 | 卖出日 | 盈亏 | 收益率 | 原因 |")
            lines.append(f"|--------|--------|------|--------|------|")
            for _, t in trades.tail(5).iterrows():
                reason = t.get("exit_reason", "signal")
                lines.append(
                    f"| {t['entry_date'].date()} | {t['exit_date'].date()} "
                    f"| {t['pnl']:+,.2f} | {t['pnl_pct']:+.1f}% | {reason} |"
                )
            lines.append(f"")

    # 横向对比
    if len(results) > 1:
        lines.append(f"## 策略对比")
        lines.append(f"")
        headers = ["策略", "收益率", "夏普", "最大回撤", "胜率", "盈亏比", "交易数"]
        lines.append(f"| {' | '.join(headers)} |")
        lines.append(f"|{'|'.join(['------'] * len(headers))}|")
        for result in results:
            m = compute_all(result)
            name = result.strategy_name.split("(")[0]
            lines.append(
                f"| {name} | {m['total_return_pct']:+.1f}% | {m['sharpe_ratio']:.2f} "
                f"| {m['max_drawdown_pct']:+.1f}% | {m['win_rate_pct']:.0f}% "
                f"| {m['profit_factor']:.2f} | {m['n_trades']} |"
            )
        lines.append(f"")

    report = "\n".join(lines)

    if save:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = _OUTPUT_DIR / f"report_{today}.md"
        path.write_text(report, encoding="utf-8")
        print(f"\n报告已保存: {path}")

    return report
