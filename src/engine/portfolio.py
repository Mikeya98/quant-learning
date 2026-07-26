"""
资产组合跟踪 —— 记录回测过程中的现金、持仓、净值变化。

核心职责：
- 记录每笔买卖后的 cash / position / avg_cost 变化
- 每日按收盘价估值（mark-to-market）
- 输出 equity curve 和 trade log

T+1 处理：
- 今日买入的股票标记为"不可卖出"
- 卖出信号触发时，跳过 T+0 的仓位
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class Trade:
    """一笔完整的交易（买入→卖出）"""
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: int = 0
    entry_cost: float = 0.0      # 买入手续费
    exit_cost: float = 0.0       # 卖出费用（佣金+印花税）
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = "signal"  # signal / stop_loss / take_profit


@dataclass
class Position:
    """当前持仓"""
    shares: int = 0
    avg_cost: float = 0.0        # 平均买入价
    buy_date: Optional[pd.Timestamp] = None  # 最近一次买入日期（T+1 用）

    @property
    def is_empty(self) -> bool:
        return self.shares == 0

    @property
    def can_sell(self) -> bool:
        """持仓非空且不是今天买入的"""
        return self.shares > 0

    def market_value(self, current_price: float) -> float:
        return self.shares * current_price


class Portfolio:
    """
    跟踪回测期间的资产状态。

    状态机：
        IDLE (无持仓) ←→ LONG (持有)

    用法:
        pf = Portfolio(initial_capital=100000)
        pf.buy(date, price, shares, commission)
        pf.update(date, close_price)   # 每日调用，记录净值
        pf.sell(date, price, shares, commission, stamp_duty)
        pf.summary()                   # 打印回测总结
    """

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = Position()

        # 历史记录
        self._equity_records: list[dict] = []
        self._trades: list[Trade] = []
        self._open_trade: Optional[Trade] = None  # 当前未平仓的交易

    # ---- 交易操作 ----

    def buy(self, date: pd.Timestamp, price: float, shares: int, commission: float):
        """
        买入股票。

        Args:
            date:       交易日期
            price:      成交价（已含滑点）
            shares:     买入股数
            commission: 手续费
        """
        cost = price * shares + commission
        if cost > self.cash:
            raise ValueError(
                f"资金不足: 需要 {cost:.2f}, 现金 {self.cash:.2f}"
            )

        self.cash -= cost
        # 计算新的平均成本
        if self.position.shares > 0:
            total_value = self.position.shares * self.position.avg_cost + price * shares
            self.position.shares += shares
            self.position.avg_cost = total_value / self.position.shares
        else:
            self.position.shares = shares
            self.position.avg_cost = price
        self.position.buy_date = date

        # 开仓记录
        if self._open_trade is None:
            self._open_trade = Trade(
                entry_date=date,
                entry_price=price,
                shares=shares,
                entry_cost=commission,
            )

    def sell(self, date: pd.Timestamp, price: float, shares: int,
             commission: float, stamp_duty: float):
        """
        卖出股票（全部或部分）。

        Args:
            date:       交易日期
            price:      成交价（已含滑点）
            shares:     卖出股数
            commission: 手续费
            stamp_duty: 印花税（仅卖出收）
        """
        if shares > self.position.shares:
            raise ValueError(
                f"持仓不足: 要卖 {shares} 股, 只有 {self.position.shares} 股"
            )

        proceeds = price * shares - commission - stamp_duty
        self.cash += proceeds
        self.position.shares -= shares

        if self.position.shares == 0:
            self.position.avg_cost = 0.0
            self.position.buy_date = None

        # 平仓记录
        if self._open_trade is not None and self.position.shares == 0:
            trade = self._open_trade
            trade.exit_date = date
            trade.exit_price = price
            trade.exit_cost = commission + stamp_duty
            trade.pnl = (price - trade.entry_price) * trade.shares - trade.entry_cost - trade.exit_cost
            trade.pnl_pct = (price / trade.entry_price - 1) * 100
            self._trades.append(trade)
            self._open_trade = None

    # ---- 每日更新 ----

    def update(self, date: pd.Timestamp, close_price: float):
        """
        每日按收盘价更新资产快照（mark-to-market）。

        Args:
            date:        日期
            close_price: 当日收盘价
        """
        position_value = self.position.market_value(close_price)
        equity = self.cash + position_value
        self._equity_records.append({
            "date": date,
            "cash": round(self.cash, 2),
            "position_value": round(position_value, 2),
            "equity": round(equity, 2),
        })

    # ---- 查询 ----

    @property
    def equity(self) -> float:
        """当前总资产 = 现金 + 持仓市值"""
        if self._equity_records:
            return self._equity_records[-1]["equity"]
        return self.initial_capital

    def equity_df(self) -> pd.DataFrame:
        """返回净值历史 DataFrame"""
        return pd.DataFrame(self._equity_records)

    def trades_df(self) -> pd.DataFrame:
        """返回交易记录 DataFrame"""
        if not self._trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "shares": t.shares,
            "pnl": round(t.pnl, 2),
            "pnl_pct": round(t.pnl_pct, 2),
            "exit_reason": t.exit_reason,
        } for t in self._trades])

    def summary(self) -> dict:
        """返回回测总结指标"""
        n_trades = len(self._trades)
        if n_trades == 0:
            return {
                "total_return": 0.0,
                "n_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
            }

        wins = sum(1 for t in self._trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in self._trades)
        final_equity = self.equity
        total_return = (final_equity / self.initial_capital - 1) * 100

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return, 2),
            "n_trades": n_trades,
            "win_rate": round(wins / n_trades * 100, 1),
            "total_pnl": round(total_pnl, 2),
        }
