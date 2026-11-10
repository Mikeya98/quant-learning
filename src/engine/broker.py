"""
模拟券商 —— 订单执行、手续费、滑点。

A 股实际规则：
- 佣金: 万分之 2.5，最低 5 元/笔，买卖都收
- 印花税: 千分之一，仅卖出时收取
- 过户费: 万分之 0.1，买卖都收（Phase 15）
- 手数: 100 股/手，委托必须是 100 的整数倍
- T+1: 今日买入，最早明日卖出（在 Portfolio 里处理）

Fill 里逐项记录 commission / stamp_duty / transfer_fee / 滑点成本，
分析层（src/analysis/costs.py）负责汇总，Broker 只管算钱。

滑点模型：
- 买入: 成交价 = 计划价 × (1 + slippage)，对买方不利
- 卖出: 成交价 = 计划价 × (1 - slippage)，对卖方不利
- 这是最简单的固定比例滑点模型，假设每次交易都按最差价格成交
"""

from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Fill:
    """成交回报"""
    side: OrderSide
    shares: int
    price: float          # 成交价（已含滑点）
    commission: float     # 佣金
    stamp_duty: float     # 印花税（仅卖出）
    total_cost: float     # 总成本（买方: 成交额+费用, 卖方: 成交额-费用）
    transfer_fee: float = 0.0   # 过户费（买卖双向，万分之 0.1）
    ref_price: float = 0.0      # 决策价（滑点前的计划价）

    @property
    def net_amount(self) -> float:
        """实际资金变动（买方为正=支出, 卖方为正=收入）"""
        if self.side == OrderSide.BUY:
            return self.total_cost
        else:
            return (self.price * self.shares - self.commission
                    - self.stamp_duty - self.transfer_fee)

    @property
    def notional(self) -> float:
        """成交额（按成交价计）"""
        return self.price * self.shares

    @property
    def explicit_cost(self) -> float:
        """显性费用：直接从现金里扣掉的佣金 + 印花税 + 过户费"""
        return self.commission + self.stamp_duty + self.transfer_fee

    @property
    def slippage_cost(self) -> float:
        """滑点成本：成交价相对决策价的价差 × 股数。

        这部分不是额外扣款，而是已经体现在成交价里 —— 同一个价格
        既算成本又算净值，所以只能作为"如果不滑点能多赚多少"的
        参考，不能和显性费用直接相加当现金支出。
        """
        if self.ref_price <= 0:
            return 0.0
        return abs(self.price - self.ref_price) * self.shares


class Broker:
    """
    模拟券商。

    接受订单 → 计算滑点 + 手续费 + 印花税 → 返回成交信息

    用法:
        broker = Broker(commission_rate=0.00025, slippage=0.001)
        fill = broker.execute_buy(price=10.0, shares=1000)
        # → Fill(side=BUY, shares=1000, price=10.01, commission=5.0, ...)
    """

    def __init__(
        self,
        commission_rate: float = 0.00025,   # 万2.5
        min_commission: float = 5.0,         # 最低佣金 5 元
        slippage: float = 0.001,            # 0.1% 滑点
        stamp_duty_rate: float = 0.001,     # 千一印花税
        transfer_fee_rate: float = 0.00001, # 万分之 0.1 过户费（双向）
        lot_size: int = 100,
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self.stamp_duty_rate = stamp_duty_rate
        self.transfer_fee_rate = transfer_fee_rate
        self.lot_size = lot_size

    # ---- 公开接口 ----

    def execute_buy(self, price: float, shares: int) -> Fill:
        """
        执行买入订单。

        Args:
            price:  计划买入价（通常为次日开盘价）
            shares: 买入股数

        Returns:
            Fill 成交信息
        """
        shares = self._round_shares(shares)
        if shares <= 0:
            raise ValueError(f"买入股数必须 > 0: {shares}")

        exec_price = self._slippage_price(price, OrderSide.BUY)
        commission = self._calc_commission(exec_price, shares)
        transfer_fee = exec_price * shares * self.transfer_fee_rate
        total_cost = exec_price * shares + commission + transfer_fee

        return Fill(
            side=OrderSide.BUY,
            shares=shares,
            price=exec_price,
            commission=commission,
            stamp_duty=0.0,   # 买入不收印花税
            total_cost=total_cost,
            transfer_fee=transfer_fee,
            ref_price=price,
        )

    def execute_sell(self, price: float, shares: int) -> Fill:
        """
        执行卖出订单。

        Args:
            price:  计划卖出价
            shares: 卖出股数

        Returns:
            Fill 成交信息
        """
        shares = self._round_shares(shares)
        if shares <= 0:
            raise ValueError(f"卖出股数必须 > 0: {shares}")

        exec_price = self._slippage_price(price, OrderSide.SELL)
        commission = self._calc_commission(exec_price, shares)
        stamp_duty = exec_price * shares * self.stamp_duty_rate
        transfer_fee = exec_price * shares * self.transfer_fee_rate

        return Fill(
            side=OrderSide.SELL,
            shares=shares,
            price=exec_price,
            commission=commission,
            stamp_duty=stamp_duty,
            total_cost=(exec_price * shares - commission
                        - stamp_duty - transfer_fee),
            transfer_fee=transfer_fee,
            ref_price=price,
        )

    def calc_shares(self, cash: float, price: float, fraction: float = 1.0) -> int:
        """
        根据可用资金计算可买入的股数。

        Args:
            cash:     可用资金
            price:    买入价
            fraction: 资金使用比例（默认 1.0 = 全仓）

        Returns:
            可买入股数（取整到手的倍数），确保不超资金
        """
        budget = cash * fraction
        effective_price = price * (1 + self.slippage)
        # 先粗算股数
        raw_shares = int(budget / effective_price)
        shares = self._round_shares(raw_shares)
        # 校验：实际买入成本是否超预算（含佣金）
        while shares > 0:
            actual_cost = effective_price * shares
            commission = self._calc_commission(effective_price, shares)
            if actual_cost + commission <= budget:
                break
            shares -= self.lot_size  # 减一手重试
        return shares

    # ---- 内部方法 ----

    def _slippage_price(self, price: float, side: OrderSide) -> float:
        """计算滑点后的成交价"""
        if side == OrderSide.BUY:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def _calc_commission(self, price: float, shares: int) -> float:
        """计算佣金（含最低 5 元限制）"""
        raw = price * shares * self.commission_rate
        return max(raw, self.min_commission)

    def _round_shares(self, shares: int) -> int:
        """取整到手的倍数（向下）"""
        return (shares // self.lot_size) * self.lot_size
