"""
多标的组合回测 —— 把单只股票的策略信号组合成一个资产组合。

和 BacktestEngine 的区别:
    BacktestEngine    一次只持有一只股票, 全仓进全仓出
    MultiAssetBacktester
                      同时持有多只股票, 每个交易日计算目标权重,
                      按目标权重调仓, 组合净值由所有持仓共同决定

组合层要处理的事情:
- 多个标的数据按日期对齐 (取交集, 避免不同上市日期带来的空档)
- 每个标的各自跑策略, 得到该标的的持仓状态 (1=持有 / 0=空仓)
- 按分配规则把资金分给"当前应该持有"的标的
- 调仓: 先卖后买, 计算手续费/滑点, 记录成交
- 每日 mark-to-market, 输出净值曲线和权重变化

分配规则:
- equal     : 所有信号为 1 的标的等权
- momentum  : 在信号为 1 的标的中, 按过去 N 日涨幅排序, 取前 top_n 只

执行模型和单标的引擎保持一致: 当日收盘信号, 当日收盘成交。
这是简化处理, 严格回测应该次日开盘成交。
"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from .broker import Broker


@dataclass
class PortfolioResult:
    """组合回测结果"""
    equity_df: pd.DataFrame          # date, cash, position_value, equity
    weights_df: pd.DataFrame         # date + 每个标的的权重
    trades_df: pd.DataFrame          # 成交记录 (每笔买卖)
    per_symbol: pd.DataFrame         # 每个标的的盈亏贡献
    correlation: pd.DataFrame        # 日收益率相关矩阵
    summary: dict
    config: dict = field(default_factory=dict)


class MultiAssetBacktester:
    """
    多标的组合回测引擎。

    用法:
        bt = MultiAssetBacktester(initial_capital=100000)
        result = bt.run(data_dict, signal_dict,
                        alloc="equal", rebalance_days=5)
        print(result.summary)
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.00025,
        slippage: float = 0.001,
        rebalance_days: int = 5,
        min_weight_change: float = 0.02,
    ):
        self.initial_capital = initial_capital
        self.broker = Broker(commission_rate=commission_rate, slippage=slippage)
        self.rebalance_days = max(1, rebalance_days)
        self.min_weight_change = min_weight_change

    # ------------------------------------------------------------------
    # 数据准备
    # ------------------------------------------------------------------

    @staticmethod
    def _align(
        data: Dict[str, pd.DataFrame],
        signals: Dict[str, pd.Series],
    ):
        """把多个标的的收盘价和信号对齐到同一组交易日。

        Returns:
            dates:    DatetimeIndex (所有标的都有数据的日子)
            prices:   DataFrame, 列=标的, 行=日期
            sig_panel: DataFrame, 列=标的, 行=日期, 值 0/1
        """
        if not data:
            raise ValueError("data 为空")

        price_series = {}
        sig_series = {}

        for symbol, df in data.items():
            if "date" not in df.columns or "close" not in df.columns:
                raise ValueError(f"{symbol} 缺少 date/close 列")

            s = df.set_index("date")["close"].sort_index()
            price_series[symbol] = s

            sig = signals.get(symbol)
            if sig is None:
                raise ValueError(f"{symbol} 没有对应信号")

            # 策略返回的 signal 是 RangeIndex, 转成日期索引再对齐
            sig = pd.Series(np.asarray(sig), index=s.index).astype(int)
            sig_series[symbol] = sig

        dates = None
        for s in price_series.values():
            dates = s.index if dates is None else dates.intersection(s.index)
        dates = dates.sort_values()

        if len(dates) == 0:
            raise ValueError("各标的数据没有重叠的交易日")

        prices = pd.DataFrame(
            {sym: s.reindex(dates) for sym, s in price_series.items()},
            index=dates,
        ).ffill()

        sig_panel = pd.DataFrame(
            {sym: s.reindex(dates).ffill().fillna(0).astype(int)
             for sym, s in sig_series.items()},
            index=dates,
        )

        return dates, prices, sig_panel

    # ------------------------------------------------------------------
    # 目标权重
    # ------------------------------------------------------------------

    def _target_weights(
        self,
        i: int,
        symbols,
        prices: pd.DataFrame,
        sig_panel: pd.DataFrame,
        alloc: str,
        top_n: int,
        momentum_window: int,
    ) -> Dict[str, float]:
        """计算第 i 个交易日的目标权重 (权重和 <= 1, 剩余为现金)"""
        active = [s for s in symbols if sig_panel[s].iloc[i] == 1]

        if alloc == "momentum" and len(active) > 1 and momentum_window > 1:
            if i >= momentum_window:
                scores = {
                    s: prices[s].iloc[i] / prices[s].iloc[i - momentum_window] - 1
                    for s in active
                }
                active = sorted(active, key=lambda s: scores[s], reverse=True)
            if top_n > 0:
                active = active[:top_n]

        if not active:
            return {}

        w = 1.0 / len(active)
        return {s: w for s in active}

    # ------------------------------------------------------------------
    # 回测主循环
    # ------------------------------------------------------------------

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        signals: Dict[str, pd.Series],
        alloc: str = "equal",
        top_n: int = 0,
        momentum_window: int = 20,
        strategy_name: str = "",
    ) -> PortfolioResult:
        dates, prices, sig_panel = self._align(data, signals)
        symbols = list(prices.columns)

        cash = self.initial_capital
        shares = {s: 0 for s in symbols}
        buy_date = {s: None for s in symbols}
        flow = {s: 0.0 for s in symbols}      # 每个标的的累计现金流 (买入为负, 卖出为正)

        equity_records = []
        weight_records = []
        trades = []

        prev_target_set = set()
        n_rebalances = 0
        traded_notional = 0.0

        for i, date in enumerate(dates):
            px = prices.iloc[i]

            # ---- 1. 用当日收盘价估值 ----
            position_value = sum(shares[s] * px[s] for s in symbols)
            equity = cash + position_value

            # ---- 2. 目标权重 ----
            targets = self._target_weights(
                i, symbols, prices, sig_panel,
                alloc=alloc, top_n=top_n, momentum_window=momentum_window,
            )
            target_set = set(targets)

            # 定期调仓, 或者持仓集合发生变化时立即调仓
            should_rebalance = (
                i == 0
                or i % self.rebalance_days == 0
                or target_set != prev_target_set
            )

            if should_rebalance and equity > 0:
                n_rebalances += 1
                target_value = {s: equity * w for s, w in targets.items()}
                threshold = self.min_weight_change * equity

                # ---- 3a. 先卖: 释放现金, 同时满足 T+1 ----
                for s in symbols:
                    cur_val = shares[s] * px[s]
                    tgt_val = target_value.get(s, 0.0)
                    delta = cur_val - tgt_val
                    exiting = s not in target_value

                    # 调仓的小差额可以忽略; 但退出组合必须清仓,
                    # 否则不到 2% 的尾巴会一直留在账上, 越积越多
                    if not exiting and delta <= threshold:
                        continue
                    if exiting and cur_val <= 0:
                        continue
                    if buy_date[s] == date:
                        continue                      # 今天买的, T+1 不能卖

                    sell_shares = int(delta / (px[s] * (1 - self.broker.slippage)))
                    sell_shares = min(sell_shares, shares[s])
                    sell_shares = (sell_shares // self.broker.lot_size) * self.broker.lot_size
                    if sell_shares <= 0:
                        continue

                    fill = self.broker.execute_sell(px[s], sell_shares)
                    cash += fill.net_amount
                    shares[s] -= sell_shares
                    flow[s] += fill.net_amount
                    traded_notional += fill.price * fill.shares

                    trades.append({
                        "date": date,
                        "symbol": s,
                        "side": "SELL",
                        "shares": fill.shares,
                        "price": round(fill.price, 4),
                        "fee": round(fill.commission + fill.stamp_duty, 2),
                    })

                # ---- 3b. 再买: 用可用现金补到目标权重 ----
                for s, tgt_val in target_value.items():
                    cur_val = shares[s] * px[s]
                    delta = tgt_val - cur_val
                    if delta <= threshold or cash <= 0:
                        continue

                    budget = min(delta, cash)
                    buy_shares = self.broker.calc_shares(budget, px[s], 1.0)
                    if buy_shares <= 0:
                        continue

                    fill = self.broker.execute_buy(px[s], buy_shares)
                    if fill.net_amount > cash:
                        continue                      # 资金不够 (最低佣金导致)

                    cash -= fill.net_amount
                    shares[s] += buy_shares
                    buy_date[s] = date
                    flow[s] -= fill.net_amount
                    traded_notional += fill.price * fill.shares

                    trades.append({
                        "date": date,
                        "symbol": s,
                        "side": "BUY",
                        "shares": fill.shares,
                        "price": round(fill.price, 4),
                        "fee": round(fill.commission, 2),
                    })

                prev_target_set = target_set

            # ---- 4. 收盘再估值一次, 记录净值与权重 ----
            position_value = sum(shares[s] * px[s] for s in symbols)
            equity = cash + position_value

            equity_records.append({
                "date": date,
                "cash": round(cash, 2),
                "position_value": round(position_value, 2),
                "equity": round(equity, 2),
            })

            weights = {"date": date}
            for s in symbols:
                weights[s] = round(shares[s] * px[s] / equity, 4) if equity > 0 else 0.0
            weight_records.append(weights)

        # ------------------------------------------------------------------
        # 结果整理
        # ------------------------------------------------------------------
        equity_df = pd.DataFrame(equity_records)
        weights_df = pd.DataFrame(weight_records).fillna(0.0)

        # 等权买入持有基准 (初始资金平均买入所有标的, 不再调仓)
        bench_series = self.initial_capital * (
            prices / prices.iloc[0]
        ).mean(axis=1)
        equity_df["benchmark"] = bench_series.values

        last_px = prices.iloc[-1]
        per_symbol_rows = []
        for s in symbols:
            last_val = shares[s] * last_px[s]
            contribution = flow[s] + last_val
            per_symbol_rows.append({
                "symbol": s,
                "final_shares": shares[s],
                "final_value": round(last_val, 2),
                "contribution": round(contribution, 2),
            })
        per_symbol = pd.DataFrame(per_symbol_rows)
        per_symbol["contribution_pct"] = (
            per_symbol["contribution"] / self.initial_capital * 100
        ).round(2)

        returns = prices.pct_change().dropna()
        correlation = returns.corr()

        equity_series = equity_df.set_index("date")["equity"]
        bench_series = equity_df.set_index("date")["benchmark"]

        summary = self._summary(
            equity_series, bench_series, correlation,
            n_rebalances=n_rebalances,
            n_trades=len(trades),
            traded_notional=traded_notional,
        )

        config = {
            "symbols": symbols,
            "strategy": strategy_name,
            "alloc": alloc,
            "top_n": top_n,
            "momentum_window": momentum_window,
            "rebalance_days": self.rebalance_days,
            "initial_capital": self.initial_capital,
        }

        return PortfolioResult(
            equity_df=equity_df,
            weights_df=weights_df,
            trades_df=pd.DataFrame(trades),
            per_symbol=per_symbol,
            correlation=correlation,
            summary=summary,
            config=config,
        )

    # ------------------------------------------------------------------
    # 绩效指标
    # ------------------------------------------------------------------

    def _summary(
        self,
        equity: pd.Series,
        benchmark: pd.Series,
        correlation: pd.DataFrame,
        n_rebalances: int,
        n_trades: int,
        traded_notional: float,
    ) -> dict:
        from ..analysis.metrics import (
            annualized_return,
            annualized_volatility,
            daily_returns,
            max_drawdown,
            sharpe_ratio,
        )

        rets = daily_returns(equity)
        dd = max_drawdown(equity)
        bench_rets = daily_returns(benchmark)

        # 相关系数矩阵去掉对角线后取平均
        if len(correlation) > 1:
            mask = ~np.eye(len(correlation), dtype=bool)
            avg_corr = float(correlation.values[mask].mean())
        else:
            avg_corr = 0.0

        return {
            "start": str(equity.index[0])[:10],
            "end": str(equity.index[-1])[:10],
            "n_days": len(equity),
            "initial_capital": round(self.initial_capital, 2),
            "final_equity": round(float(equity.iloc[-1]), 2),
            "total_return_pct": round(
                (equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
            "annual_return_pct": round(annualized_return(equity), 2),
            "annual_volatility_pct": round(annualized_volatility(rets), 2),
            "sharpe_ratio": round(sharpe_ratio(equity), 2),
            "max_drawdown_pct": dd["max_dd"],
            "max_dd_duration_days": dd["duration_days"],
            "benchmark_return_pct": round(
                (benchmark.iloc[-1] / benchmark.iloc[0] - 1) * 100, 2),
            "benchmark_volatility_pct": round(annualized_volatility(bench_rets), 2),
            "excess_return_pct": round(
                (equity.iloc[-1] / equity.iloc[0]
                 - benchmark.iloc[-1] / benchmark.iloc[0]) * 100, 2),
            "avg_pairwise_corr": round(avg_corr, 3),
            "n_rebalances": n_rebalances,
            "n_trades": n_trades,
            "turnover_x": round(traded_notional / equity.mean(), 2),
        }


def format_report(result: PortfolioResult) -> str:
    """把组合回测结果格式化成终端报告"""
    s = result.summary
    cfg = result.config

    lines = []
    lines.append("=" * 62)
    lines.append("  组合回测结果")
    lines.append("=" * 62)
    lines.append(f"  区间:       {s['start']} ~ {s['end']}  ({s['n_days']} 个交易日)")
    lines.append(f"  标的:       {', '.join(cfg['symbols'])}")
    lines.append(f"  策略:       {cfg['strategy']}")
    lines.append(f"  分配规则:   {cfg['alloc']}"
                 + (f" (top {cfg['top_n']})" if cfg['alloc'] == 'momentum' else ""))
    lines.append(f"  调仓周期:   {cfg['rebalance_days']} 个交易日")
    lines.append("")
    lines.append(f"  初始资金:   {s['initial_capital']:>14,.2f}")
    lines.append(f"  最终净值:   {s['final_equity']:>14,.2f}")
    lines.append(f"  总收益率:   {s['total_return_pct']:>+13.2f}%")
    lines.append(f"  年化收益:   {s['annual_return_pct']:>+13.2f}%")
    lines.append(f"  年化波动:   {s['annual_volatility_pct']:>13.2f}%")
    lines.append(f"  夏普比率:   {s['sharpe_ratio']:>14.2f}")
    lines.append(f"  最大回撤:   {s['max_drawdown_pct']:>+13.2f}%")
    lines.append(f"  回撤持续:   {s['max_dd_duration_days']:>13d} 天")
    lines.append("")
    lines.append(f"  等权基准收益: {s['benchmark_return_pct']:>+11.2f}%")
    lines.append(f"  超额收益:     {s['excess_return_pct']:>+11.2f}%")
    lines.append(f"  平均相关系数: {s['avg_pairwise_corr']:>11.3f}")
    lines.append("")
    lines.append(f"  调仓次数:   {s['n_rebalances']}")
    lines.append(f"  成交笔数:   {s['n_trades']}")
    lines.append(f"  换手倍数:   {s['turnover_x']:.2f}x (累计成交额 / 平均净值)")

    lines.append("")
    lines.append("-" * 62)
    lines.append("  各标的贡献 (期末市值 + 累计现金流)")
    lines.append("-" * 62)
    lines.append(f"  {'标的':<10}{'期末市值':>14}{'贡献':>14}{'占比':>10}")
    for _, row in result.per_symbol.sort_values(
            "contribution", ascending=False).iterrows():
        lines.append(
            f"  {row['symbol']:<10}{row['final_value']:>14,.2f}"
            f"{row['contribution']:>14,.2f}{row['contribution_pct']:>9.2f}%"
        )

    return "\n".join(lines)
