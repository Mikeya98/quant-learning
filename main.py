#!/usr/bin/env python3
"""
Quant Learning —— 迷你 A 股回测框架

一个用于学习量化交易策略的渐进式项目。
从数据获取开始，逐步构建完整的回测系统。

用法:
    # 下载股票数据
    python main.py fetch 000001
    python main.py fetch 600519 --start 20200101 --end 20241231

    # 查看已下载的数据
    python main.py list

    # 查看单只股票概况
    python main.py info 000001
"""

import argparse
import sys
import os

# 确保 src 在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fetcher import fetch_stock_daily
from src.data.cleaner import clean_data, add_returns
from src.data.storage import save_csv, load_csv, list_stocks
from src.strategy import STRATEGY_REGISTRY
from src.engine import BacktestEngine
from src.analysis import compute_all, plot_equity_curve, plot_trade_signals
from src.analysis.optimizer import grid_search as gs, walk_forward as wf
from src.risk import StopManager


def cmd_fetch(args):
    """下载股票数据"""
    df = fetch_stock_daily(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        adjust=args.adjust,
    )
    # 保存原始数据
    save_csv(df, args.symbol, processed=False)

    # 清洗并保存
    df_clean = clean_data(df)
    df_clean = add_returns(df_clean)
    save_csv(df_clean, args.symbol, processed=True)

    # 简要预览
    print(f"\n数据预览 ({args.symbol}):")
    print(f"  最早日期: {df_clean['date'].min().date()}")
    print(f"  最晚日期: {df_clean['date'].max().date()}")
    print(f"  停牌天数: {df_clean['suspended'].sum()}")
    print(f"  累计收益: {df_clean['cum_ret'].iloc[-1]:.2%}")


def cmd_list(args):
    """列出已下载的股票"""
    raw = list_stocks(processed=False)
    proc = list_stocks(processed=True)
    print("已下载的股票:")
    print(f"  原始数据 ({len(raw)} 只): {', '.join(raw) if raw else '(无)'}")
    print(f"  清洗数据 ({len(proc)} 只): {', '.join(proc) if proc else '(无)'}")


def cmd_info(args):
    """查看单只股票概况"""
    df = load_csv(args.symbol, processed=True)
    print(f"\n===== {args.symbol} 数据概况 =====")
    print(f"日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"交易日数: {len(df)}")
    print(f"停牌天数: {df['suspended'].sum()}")
    print(f"\n价格统计:")
    print(f"  收盘价: 最低 {df['close'].min():.2f} / 最高 {df['close'].max():.2f} / 最新 {df['close'].iloc[-1]:.2f}")
    print(f"  日均成交量: {df['volume'].mean():,.0f} 手")
    print(f"\n收益统计:")
    print(f"  日均收益率: {df['ret'].mean():.4%}")
    print(f"  日收益率标准差: {df['ret'].std():.4%}")
    print(f"  累计收益: {df['cum_ret'].iloc[-1]:.2%}")
    print(f"  最大单日涨幅: {df['ret'].max():.2%}")
    print(f"  最大单日跌幅: {df['ret'].min():.2%}")


def cmd_signal(args):
    """运行策略并显示信号统计"""
    df = load_csv(args.symbol, processed=True)

    # 查找策略类
    strategy_cls = STRATEGY_REGISTRY.get(args.strategy)
    if strategy_cls is None:
        print(f"未知策略: {args.strategy}")
        print(f"可用策略: {', '.join(STRATEGY_REGISTRY.keys())}")
        return

    # 解析额外参数（如 --param fast=5 --param slow=20）
    params = {}
    if args.param:
        for p in args.param:
            key, val = p.split("=")
            # 自动转换数值参数
            try:
                val = float(val)
                if val == int(val):
                    val = int(val)
            except ValueError:
                pass
            params[key] = val

    # 实例化策略并运行
    strategy = strategy_cls(**params)
    print(f"\n策略: {strategy.describe()}")
    print(f"标的: {args.symbol}")

    signals = strategy.run(df)

    # 信号统计
    stats = strategy.signal_stats()
    print(f"\n===== 信号统计 =====")
    print(f"  总 bar 数: {stats['total_bars']}")
    print(f"  买入信号: {stats['buy_signals']} ({stats['buy_ratio']})")
    print(f"  卖出信号: {stats['sell_signals']}")
    print(f"  完整交易对: {stats['trade_count']}")

    # 最近几次交易信号
    trade_dates = df["date"][signals != 0]
    print(f"\n最近 10 个交易日信号:")
    recent = df[signals != 0].tail(10)
    for _, row in recent.iterrows():
        sig = "买入 ▲" if signals[row.name] == 1 else "卖出 ▼"
        print(f"  {row['date'].date()}  {sig}  收盘价: {row['close']:.2f}")


def cmd_backtest(args):
    """运行回测"""
    df = load_csv(args.symbol, processed=True)

    strategy_cls = STRATEGY_REGISTRY.get(args.strategy)
    if strategy_cls is None:
        print(f"未知策略: {args.strategy}")
        print(f"可用策略: {', '.join(STRATEGY_REGISTRY.keys())}")
        return

    # 解析策略参数
    params = {}
    if args.param:
        for p in args.param:
            key, val = p.split("=")
            try:
                val = float(val)
                if val == int(val):
                    val = int(val)
            except ValueError:
                pass
            params[key] = val

    # 生成信号
    strategy = strategy_cls(**params)
    signals = strategy.run(df)

    # 止损止盈设置
    stop_mgr = StopManager(
        stop_loss_pct=args.stop_loss / 100 if args.stop_loss else 0,
        take_profit_pct=args.take_profit / 100 if args.take_profit else 0,
    )

    # 运行回测
    engine = BacktestEngine(
        initial_capital=args.capital,
        position_pct=args.position,
        stop_manager=stop_mgr,
    )
    result = engine.run(
        df, signals,
        symbol=args.symbol,
        strategy_name=strategy.describe(),
    )

    # 输出结果
    s = result.summary
    print(f"\n{'='*50}")
    print(f"  回测结果")
    print(f"{'='*50}")
    print(f"  策略: {result.strategy_name}")
    print(f"  标的: {result.symbol}")
    print(f"  初始资金: {s['initial_capital']:,.2f}")
    print(f"  最终净值: {s['final_equity']:,.2f}")
    print(f"  总收益率: {s['total_return_pct']:+.2f}%")
    print(f"  交易次数: {s['n_trades']}")
    print(f"  胜率:     {s['win_rate']:.1f}%")
    print(f"  总盈亏:   {s['total_pnl']:+,.2f}")

    # 最近几笔交易
    trades = result.trades_df
    if len(trades) > 0:
        print(f"\n  最近 5 笔交易:")
        for _, t in trades.tail(5).iterrows():
            print(f"    {t['entry_date'].date()} → {t['exit_date'].date()}  "
                  f"{t['pnl']:+,.2f}  ({t['pnl_pct']:+.1f}%)")

    # 完整绩效分析
    if args.analyze:
        metrics = compute_all(result)
        print(f"\n{'='*50}")
        print(f"  绩效指标")
        print(f"{'='*50}")
        print(f"  年化收益率:   {metrics['annual_return_pct']:+.2f}%")
        print(f"  年化波动率:   {metrics['annual_volatility_pct']:.2f}%")
        print(f"  夏普比率:     {metrics['sharpe_ratio']:.2f}")
        print(f"  最大回撤:     {metrics['max_drawdown_pct']:+.1f}%")
        print(f"  回撤持续:     {metrics['max_dd_duration_days']} 天")
        print(f"  Calmar 比率:  {metrics['calmar_ratio']:.2f}")
        print(f"  盈亏比:       {metrics['profit_factor']:.2f}")
        print(f"  平均盈亏:     {metrics['avg_trade_pnl_pct']:+.2f}%")

        chart1 = plot_equity_curve(
            result.equity_df, df, title=f"{result.strategy_name} @ {result.symbol}"
        )
        chart2 = plot_trade_signals(
            df, result.trades_df,
            title=f"{result.strategy_name} @ {result.symbol}"
        )
        print(f"\n  图表已保存: {chart1}")
        print(f"  图表已保存: {chart2}")


def cmd_optimize(args):
    """参数优化"""
    df = load_csv(args.symbol, processed=True)

    strategy_cls = STRATEGY_REGISTRY.get(args.strategy)
    if strategy_cls is None:
        print(f"未知策略: {args.strategy}")
        return

    # 解析参数范围: "fast=5,10,15;slow=20,30,60"
    param_grid = {}
    for part in args.params.split(";"):
        key, vals = part.split("=")
        param_grid[key.strip()] = [int(v) for v in vals.split(",")]

    if args.mode == "grid":
        print(f"\n网格搜索: {strategy_cls.__name__} @ {args.symbol}")
        print(f"参数空间: {param_grid}")

        result_df = gs(df, strategy_cls, param_grid, metric="sharpe")

        if len(result_df) == 0:
            print("无有效结果")
            return

        print(f"\n{'='*70}")
        print(f"  Top 10 参数组合（按夏普排序）")
        print(f"{'='*70}")
        cols = list(param_grid.keys()) + ["total_return_pct", "sharpe", "max_dd_pct", "n_trades"]
        print(result_df[cols].head(10).to_string(index=False))

    elif args.mode == "walkforward":
        print(f"\n滚动优化: {strategy_cls.__name__} @ {args.symbol}")
        print(f"参数空间: {param_grid}")

        result_df = wf(df, strategy_cls, param_grid, metric="sharpe")

        if len(result_df) == 0:
            print("数据不足，无法执行滚动优化")
            return

        print(f"\n{'='*90}")
        print(f"  滚动窗口结果")
        print(f"{'='*90}")
        show_cols = ["fold", "test_period"] + [f"best_{k}" for k in param_grid] + \
                    ["train_sharpe", "test_sharpe", "test_return", "test_max_dd"]
        available = [c for c in show_cols if c in result_df.columns]
        print(result_df[available].to_string(index=False))
        metrics = compute_all(result)
        print(f"\n{'='*50}")
        print(f"  绩效指标")
        print(f"{'='*50}")
        print(f"  年化收益率:   {metrics['annual_return_pct']:+.2f}%")
        print(f"  年化波动率:   {metrics['annual_volatility_pct']:.2f}%")
        print(f"  夏普比率:     {metrics['sharpe_ratio']:.2f}")
        print(f"  最大回撤:     {metrics['max_drawdown_pct']:+.1f}%")
        print(f"  回撤持续:     {metrics['max_dd_duration_days']} 天")
        print(f"  Calmar 比率:  {metrics['calmar_ratio']:.2f}")
        print(f"  盈亏比:       {metrics['profit_factor']:.2f}")
        print(f"  平均盈亏:     {metrics['avg_trade_pnl_pct']:+.2f}%")

        # 生成图表
        chart1 = plot_equity_curve(
            result.equity_df, df, title=f"{result.strategy_name} @ {result.symbol}"
        )
        chart2 = plot_trade_signals(
            df, result.trades_df,
            title=f"{result.strategy_name} @ {result.symbol}"
        )
        print(f"\n  图表已保存: {chart1}")
        print(f"  图表已保存: {chart2}")


def main():
    parser = argparse.ArgumentParser(
        description="Quant Learning - 迷你A股回测框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py fetch 000001            # 下载平安银行全部历史数据
  python main.py fetch 600519 --start 20230101  # 下载茅台从2023年起的数据
  python main.py list                    # 列出已下载的股票
  python main.py info 000001             # 查看平安银行数据概况
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # fetch 命令
    p_fetch = subparsers.add_parser("fetch", help="下载股票数据")
    p_fetch.add_argument("symbol", help="股票代码（如 000001, 600519）")
    p_fetch.add_argument("--start", default="20100101", help="起始日期 YYYYMMDD（默认 20100101）")
    p_fetch.add_argument("--end", default="20251231", help="结束日期 YYYYMMDD（默认 20251231）")
    p_fetch.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", ""],
                          help="复权方式: qfq=前复权(默认), hfq=后复权, 空=不复权")

    # list 命令
    subparsers.add_parser("list", help="列出已下载的股票")

    # info 命令
    p_info = subparsers.add_parser("info", help="查看股票数据概况")
    p_info.add_argument("symbol", help="股票代码")

    # signal 命令
    p_signal = subparsers.add_parser("signal", help="运行策略查看信号")
    p_signal.add_argument("strategy", choices=list(STRATEGY_REGISTRY.keys()),
                          help="策略名称")
    p_signal.add_argument("symbol", help="股票代码")
    p_signal.add_argument("--param", "-p", action="append",
                          help="策略参数，格式 key=value（可多次使用）")

    # backtest 命令
    p_bt = subparsers.add_parser("backtest", help="运行回测")
    p_bt.add_argument("strategy", choices=list(STRATEGY_REGISTRY.keys()),
                      help="策略名称")
    p_bt.add_argument("symbol", help="股票代码")
    p_bt.add_argument("--param", "-p", action="append",
                      help="策略参数，格式 key=value")
    p_bt.add_argument("--capital", type=float, default=100000,
                      help="初始资金（默认 100000）")
    p_bt.add_argument("--position", type=float, default=1.0,
                      help="仓位比例（默认 1.0 = 全仓）")
    p_bt.add_argument("--analyze", "-a", action="store_true",
                      help="计算完整绩效指标并生成图表")
    p_bt.add_argument("--stop-loss", type=float, default=0,
                      help="止损比例（%%，如 5 = 亏5%%平仓）")
    p_bt.add_argument("--take-profit", type=float, default=0,
                      help="止盈比例（%%，如 15 = 赚15%%平仓）")

    # optimize 命令
    p_opt = subparsers.add_parser("optimize", help="参数优化")
    p_opt.add_argument("mode", choices=["grid", "walkforward"],
                       help="grid=网格搜索, walkforward=滚动优化")
    p_opt.add_argument("strategy", choices=list(STRATEGY_REGISTRY.keys()),
                       help="策略名称")
    p_opt.add_argument("symbol", help="股票代码")
    p_opt.add_argument("--params", "-p", required=True,
                       help="参数范围，格式 key=1,2,3;key2=10,20，分号分隔")

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "signal":
        cmd_signal(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
