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

import pandas as pd

# 确保 src 在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fetcher import fetch_stock_daily
from src.data.cleaner import clean_data, add_returns
from src.data.storage import save_csv, load_csv, list_stocks
from src.strategy import STRATEGY_REGISTRY
from src.engine import BacktestEngine, MultiAssetBacktester, format_report
from src.analysis import (
    compute_all, plot_equity_curve, plot_trade_signals,
    plot_portfolio, plot_correlation,
)
from src.analysis.optimizer import grid_search as gs, walk_forward as wf
from src.risk import StopManager
from src.analysis.report import generate_report
from src.analysis.factor_research import (
    compute_forward_returns, compute_ic, quantile_returns, quantile_summary,
    make_factors, format_ic_report,
)
from src.analysis.factor_visualize import plot_ic, plot_quantiles
from src.analysis.neutralize import (
    neutralize_factor, size_exposure, industry_exposure,
)
from src.data.meta import load_industry, float_mktcap, log_size


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
    print(f"  累计收益: {df_clean['cum_ret'].iloc[-1] - 1:+.2%}")


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
    print(f"  累计收益: {df['cum_ret'].iloc[-1] - 1:+.2%}")
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


def cmd_compare(args):
    """多策略对比"""
    strategies = args.strategies.split(",")
    df = load_csv(args.symbol, processed=True)
    results = []

    print(f"\n多策略对比: {args.symbol}")
    print(f"{'='*60}")

    for sname in strategies:
        sname = sname.strip()
        strategy_cls = STRATEGY_REGISTRY.get(sname)
        if strategy_cls is None:
            print(f"  未知策略: {sname}，跳过")
            continue

        # 解析参数
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

        strategy = strategy_cls(**params)
        signals = strategy.run(df)
        engine = BacktestEngine(initial_capital=args.capital)
        result = engine.run(df, signals, symbol=args.symbol, strategy_name=strategy.describe())
        results.append(result)

        m = result.summary
        print(f"  {strategy.describe():40s}  {m['total_return_pct']:+7.2f}%  "
              f"胜率:{m['win_rate']:5.1f}%  交易:{m['n_trades']:2d}次")

    if len(results) > 1:
        report = generate_report(results, save=True)
        print(report)


def _parse_params(param_list):
    """解析 --param key=value（可多次使用）"""
    params = {}
    if not param_list:
        return params
    for p in param_list:
        key, val = p.split("=")
        try:
            val = float(val)
            if val == int(val):
                val = int(val)
        except ValueError:
            pass
        params[key] = val
    return params


def cmd_fetch_all(args):
    """批量下载多只股票"""
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"\n批量下载 {len(symbols)} 只股票: {', '.join(symbols)}")

    ok, failed = [], []
    for symbol in symbols:
        try:
            df = fetch_stock_daily(
                symbol=symbol, start_date=args.start,
                end_date=args.end, adjust=args.adjust,
            )
            save_csv(df, symbol, processed=False)
            df = add_returns(clean_data(df))
            save_csv(df, symbol, processed=True)
            ok.append(symbol)
        except Exception as e:
            print(f"  [FAIL] {symbol}: {e}")
            failed.append(symbol)

    print(f"\n完成: 成功 {len(ok)} 只, 失败 {len(failed)} 只")
    if failed:
        print(f"  失败: {', '.join(failed)}")


def cmd_universe(args):
    """选股票池：全市场快照 → 按流通市值取前 N → 可选下载历史"""
    from src.data.universe import build_universe, save_universe
    from src.data.meta import save_industry

    print(f"\n选池: 全市场快照 → 流通市值前 {args.top} 只")
    try:
        pool = build_universe(top_n=args.top)
    except Exception as e:                       # noqa: BLE001
        print(f"[FAIL] 选池失败: {e}")
        print("  接口限流时可以隔几分钟再试；已有股票池文件仍然可用。")
        return

    path = save_universe(pool)
    print(f"[universe] 已保存: {path} ({len(pool)} 只)")

    show = pool.head(20)
    print(f"\n{'代码':<8}{'名称':<10}{'行业':<12}{'流通市值(亿)':>12}")
    print("-" * 44)
    for _, row in show.iterrows():
        cap = row["float_mktcap"] / 1e8 if pd.notna(row["float_mktcap"]) else 0
        print(f"{row['symbol']:<8}{row['name']:<10}"
              f"{str(row['industry'] or '-'):<12}{cap:>12.0f}")
    if len(pool) > len(show):
        print(f"... 其余 {len(pool) - len(show)} 只略")

    # 行业信息顺手合并进行业表
    if pool["industry"].notna().any():
        ind = pool.set_index("symbol")["industry"].dropna()
        names = pool.set_index("symbol")["name"]
        p = save_industry(ind, names=names)
        print(f"[universe] 行业表已更新: {p}")

    if not args.fetch:
        print("\n提示: 加 --fetch 可以顺便下载这些股票的历史行情")
        return

    symbols = list(pool["symbol"])
    print(f"\n开始下载 {len(symbols)} 只股票的历史...")
    ok, failed = [], []
    for i, symbol in enumerate(symbols, 1):
        try:
            df = fetch_stock_daily(symbol=symbol, start_date=args.start,
                                   end_date=args.end, adjust=args.adjust)
            save_csv(df, symbol, processed=False)
            save_csv(add_returns(clean_data(df)), symbol, processed=True)
            ok.append(symbol)
        except Exception as e:                   # noqa: BLE001
            print(f"  [FAIL] {symbol}: {e}")
            failed.append(symbol)
        if i % 10 == 0:
            print(f"  进度 {i}/{len(symbols)}")

    print(f"\n完成: 成功 {len(ok)} 只, 失败 {len(failed)} 只")
    if ok:
        print("  下一步: python main.py factor --symbols "
              + ",".join(ok[:20]) + ("... " if len(ok) > 20 else ""))


def cmd_portfolio(args):
    """多标的组合回测"""
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if len(symbols) < 2:
        print("组合回测至少需要 2 只股票")
        return

    strategy_cls = STRATEGY_REGISTRY.get(args.strategy)
    if strategy_cls is None:
        print(f"未知策略: {args.strategy}")
        print(f"可用策略: {', '.join(STRATEGY_REGISTRY.keys())}")
        return

    params = _parse_params(args.param)
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

    data, signals = {}, {}
    missing = []

    for symbol in symbols:
        try:
            df = load_csv(symbol, processed=True)
        except FileNotFoundError:
            missing.append(symbol)
            continue

        df = df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)
        if len(df) < 60:
            print(f"[WARN] {symbol} 在区间内只有 {len(df)} 条数据")
            continue

        data[symbol] = df
        strategy = strategy_cls(**params)
        signals[symbol] = strategy.run(df)

    if missing:
        print(f"\n缺少数据: {', '.join(missing)}")
        print(f"先运行: python main.py fetch-all {','.join(missing)} "
              f"--start {args.start} --end {args.end}")
        return

    if len(data) < 2:
        print("可用标的不够, 无法组合回测")
        return

    strategy_name = strategy_cls(**params).describe()

    print(f"\n组合回测: {len(data)} 只标的, {strategy_name}")
    print(f"区间: {args.start} ~ {args.end}, 调仓周期: {args.rebalance} 日")

    engine = MultiAssetBacktester(
        initial_capital=args.capital,
        rebalance_days=args.rebalance,
    )
    result = engine.run(
        data, signals,
        alloc=args.alloc,
        top_n=args.top,
        momentum_window=args.momentum_window,
        strategy_name=strategy_name,
    )

    print("\n" + format_report(result))

    print("\n" + "-" * 62)
    print("  日收益率相关矩阵")
    print("-" * 62)
    print(result.correlation.round(2).to_string())

    if args.analyze:
        for path in plot_portfolio(result):
            print(f"\n  图表已保存: {path}")
        path = plot_correlation(result)
        if path:
            print(f"  图表已保存: {path}")


def _load_universe(symbols, start, end):
    """把多只股票的 CSV 拼成宽表 (index=日期, columns=代码)"""
    fields = {"close": {}, "volume": {}, "turnover": {}, "amount": {}}
    missing = []

    for symbol in symbols:
        try:
            df = load_csv(symbol, processed=True)
        except FileNotFoundError:
            missing.append(symbol)
            continue

        df = df[(df["date"] >= start) & (df["date"] <= end)]
        if df.empty:
            missing.append(symbol)
            continue

        df = df.set_index("date")
        for name in fields:
            if name in df.columns:
                fields[name][symbol] = df[name]

    if missing:
        print(f"[WARN] 跳过无数据的标的: {', '.join(missing)}")

    close = pd.DataFrame(fields["close"]).sort_index()
    volume = pd.DataFrame(fields["volume"]).sort_index() if fields["volume"] else None
    turnover = pd.DataFrame(fields["turnover"]).sort_index() if fields["turnover"] else None
    amount = pd.DataFrame(fields["amount"]).sort_index() if fields["amount"] else None
    return close, volume, turnover, amount


def cmd_factor(args):
    """因子 IC 分析与分层回测"""
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    start = pd.to_datetime(args.start)
    end = pd.to_datetime(args.end)

    close, volume, turnover, amount = _load_universe(symbols, start, end)
    if close.shape[1] < 5:
        print(f"可用标的只有 {close.shape[1]} 只, IC 分析至少需要 5 只")
        return

    print(f"\n因子研究: {close.shape[1]} 只标的, "
          f"{close.index[0].date()} ~ {close.index[-1].date()}, "
          f"持有期 {args.period} 日")

    factors = make_factors(close, volume, turnover)
    if args.factors:
        wanted = [f.strip() for f in args.factors.split(",")]
        unknown = [f for f in wanted if f not in factors]
        if unknown:
            print(f"未知因子: {', '.join(unknown)}")
            print(f"可用因子: {', '.join(factors.keys())}")
            return
        factors = {k: factors[k] for k in wanted}

    fwd = compute_forward_returns(close, periods=args.period)

    # ---------------- 中性化准备（v0.9 Phase 9） ----------------
    industry = None
    logcap_s = None
    if args.neutralize:
        parts = [p.strip() for p in args.neutralize.split(",") if p.strip()]
        unknown = [p for p in parts if p not in ("industry", "size")]
        if unknown:
            print(f"未知的中性化项: {', '.join(unknown)}（可选 industry,size）")
            return

        if "industry" in parts:
            industry = load_industry(args.industry_file)
            covered = industry.reindex(close.columns).notna().sum()
            print(f"[中性化] 行业分类: 覆盖 {covered}/{close.shape[1]} 只标的"
                  f"（{args.industry_file or 'data/meta/industry.csv'}）")
            if covered == 0:
                print("[中性化] 行业表为空, 跳过行业中性化")
                parts.remove("industry")

        if "size" in parts:
            if amount is None or turnover is None:
                print("[中性化] 缺成交额/换手率字段, 跳过市值中性化")
            else:
                mktcap = float_mktcap(amount, turnover)
                logcap_s = log_size(mktcap)
                valid = logcap_s.notna().sum().sum()
                print(f"[中性化] 市值由 成交额/换手率 反推, 有效格点 {valid}")

    results = {}
    qsummaries = {}
    for name, fac in factors.items():
        r = compute_ic(fac, fwd, method=args.method)
        r.periods = args.period
        results[name] = r
        if args.quantiles:
            qret = quantile_returns(fac, fwd, n_quantiles=args.quantiles)
            qsummaries[name] = quantile_summary(
                qret, holding_period=args.period)

    print("\n" + format_ic_report(results, qsummaries))

    # ---------------- 中性化前后对比 ----------------
    if args.neutralize and (industry is not None or logcap_s is not None):
        print("\n" + "=" * 62)
        print("  中性化前后 IC 对比")
        print("=" * 62)
        print(f"{'因子':<16}{'原始 IC':>10}{'中性 IC':>10}"
              f"{'原始市值暴露':>14}{'中性市值暴露':>14}")
        print("-" * 62)

        for name, fac in factors.items():
            raw_exp = (size_exposure(fac, logcap_s)
                       if logcap_s is not None else float("nan"))
            neu = neutralize_factor(
                fac, industry=industry, log_mktcap=logcap_s,
                winsorize=(args.winsor_lower, args.winsor_upper))
            neu_exp = (size_exposure(neu, logcap_s)
                       if logcap_s is not None else float("nan"))
            rn = compute_ic(neu, fwd, method=args.method)
            print(f"{name:<16}{results[name].mean_ic:>10.4f}"
                  f"{rn.mean_ic:>10.4f}{raw_exp:>14.0%}{neu_exp:>14.0%}")

        print("-" * 62)
        print("  市值暴露 = 因子与 ln(流通市值) 的横截面相关系数日均值")
        print("  中性化后应该接近 0; IC 掉得越多, 说明原来的 IC 越依赖行业/市值")

        if industry is not None:
            print("\n  中性化后行业暴露（行业内平均 z 值, 应接近 0）:")
            neu_all = neutralize_factor(
                factors[next(iter(factors))], industry=industry,
                log_mktcap=logcap_s,
                winsorize=(args.winsor_lower, args.winsor_upper))
            exp = industry_exposure(neu_all, industry)
            for ind_name, val in exp.items():
                print(f"    {ind_name:<10}{val:+.3f}")

    if args.analyze:
        for name, r in results.items():
            for path in plot_ic(r, holdings=args.period, title=name):
                print(f"\n  图表已保存: {path}")
            if args.quantiles:
                qret = quantile_returns(factors[name], fwd, args.quantiles)
                for path in plot_quantiles(qret, holdings=args.period, title=name):
                    print(f"  图表已保存: {path}")



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

    # compare 命令
    p_cmp = subparsers.add_parser("compare", help="多策略对比")
    p_cmp.add_argument("symbol", help="股票代码")
    p_cmp.add_argument("--strategies", "-s", required=True,
                       help="策略列表，逗号分隔，如 ma_cross,macd,rsi")
    p_cmp.add_argument("--param", "-p", action="append",
                       help="策略参数（所有策略共用）")
    p_cmp.add_argument("--capital", type=float, default=100000,
                       help="初始资金")

    # fetch-all 命令
    p_fa = subparsers.add_parser("fetch-all", help="批量下载多只股票")
    p_fa.add_argument("symbols", help="股票代码列表，逗号分隔")
    p_fa.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD")
    p_fa.add_argument("--end", default="20241231", help="结束日期 YYYYMMDD")
    p_fa.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", ""],
                      help="复权方式: qfq=前复权(默认)")

    # portfolio 命令
    p_pf = subparsers.add_parser("portfolio", help="多标的组合回测")
    p_pf.add_argument("--symbols", required=True,
                      help="股票代码列表，逗号分隔")
    p_pf.add_argument("--strategy", default="ma_cross",
                      choices=list(STRATEGY_REGISTRY.keys()),
                      help="每个标的使用的策略")
    p_pf.add_argument("--param", "-p", action="append",
                      help="策略参数，格式 key=value")
    p_pf.add_argument("--alloc", choices=["equal", "momentum"], default="equal",
                      help="equal=等权, momentum=动量排序取前 N")
    p_pf.add_argument("--top", type=int, default=3,
                      help="momentum 模式持有几只（默认 3）")
    p_pf.add_argument("--momentum-window", type=int, default=20,
                      help="动量回看天数（默认 20）")
    p_pf.add_argument("--rebalance", type=int, default=5,
                      help="调仓周期，交易日（默认 5）")
    p_pf.add_argument("--capital", type=float, default=100000,
                      help="初始资金（默认 100000）")
    p_pf.add_argument("--start", default="20190101",
                      help="起始日期 YYYYMMDD")
    p_pf.add_argument("--end", default="20241231",
                      help="结束日期 YYYYMMDD")
    p_pf.add_argument("--analyze", "-a", action="store_true",
                      help="生成净值和权重图表")

    # factor 命令
    p_fa2 = subparsers.add_parser("factor", help="因子 IC 分析与分层回测")
    p_fa2.add_argument("--symbols", required=True,
                       help="股票代码列表，逗号分隔")
    p_fa2.add_argument("--factors", default="",
                       help="要分析的因子，逗号分隔（默认全部）")
    p_fa2.add_argument("--period", type=int, default=5,
                       help="未来收益的持有期，交易日（默认 5）")
    p_fa2.add_argument("--method", choices=["spearman", "pearson"],
                       default="spearman", help="相关系数类型（默认 rank IC）")
    p_fa2.add_argument("--quantiles", type=int, default=5,
                       help="分层数量（默认 5，0 表示不分层）")
    p_fa2.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD")
    p_fa2.add_argument("--end", default="20241231", help="结束日期 YYYYMMDD")
    p_fa2.add_argument("--neutralize", default="",
                       help="中性化: industry,size (逗号分隔, 默认不做)")
    p_fa2.add_argument("--industry-file", default=None,
                       help="行业分类 CSV (默认 data/meta/industry.csv)")
    p_fa2.add_argument("--winsor-lower", type=float, default=0.01,
                       help="缩尾下分位 (默认 0.01)")
    p_fa2.add_argument("--winsor-upper", type=float, default=0.99,
                       help="缩尾上分位 (默认 0.99)")
    p_fa2.add_argument("--analyze", "-a", action="store_true",
                       help="生成 IC 和分层图表")

    # universe 命令
    p_univ = subparsers.add_parser("universe", help="选股票池（按流通市值前 N）")
    p_univ.add_argument("--top", type=int, default=50,
                        help="取前多少只（默认 50）")
    p_univ.add_argument("--fetch", action="store_true",
                        help="顺便下载这些股票的历史行情")
    p_univ.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD")
    p_univ.add_argument("--end", default="20241231", help="结束日期 YYYYMMDD")
    p_univ.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", ""])

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
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "fetch-all":
        cmd_fetch_all(args)
    elif args.command == "portfolio":
        cmd_portfolio(args)
    elif args.command == "factor":
        cmd_factor(args)
    elif args.command == "universe":
        cmd_universe(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
