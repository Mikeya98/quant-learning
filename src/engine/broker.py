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

两种模型（Phase 16）：
- fixed: 固定比例滑点，假设每次交易都按最差价格成交
- sqrt:  固定滑点 + 平方根冲击成本

    冲击 = impact_coeff × sqrt(参与率)
    参与率 = 本笔成交额 / 当日成交额

  平方根模型来自 Almgren 的市场冲击经验规律：订单越大，单位
  冲击越高，但不是线性的。参与率 10%、系数 1% 时冲击约 31.6 bps。
  成交额缺失（NaN）时按 0 参与率处理，退化成固定滑点。
"""

import math
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
    participation: float = 0.0  # 本笔成交额 / 当日成交额（Phase 16）
    impact_rate: float = 0.0    # 平方根冲击成本（比例，不含固定滑点）

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
        slippage: float = 0.001,            # 0.1% 固定滑点
        stamp_duty_rate: float = 0.001,     # 千一印花税
        transfer_fee_rate: float = 0.00001, # 万分之 0.1 过户费（双向）
        lot_size: int = 100,
        slippage_model: str = "fixed",      # fixed | sqrt
        impact_coeff: float = 0.01,         # 平方根模型系数（1% @ 100% 参与率）
        max_impact: float = 0.05,           # 冲击成本上限（5%）
    ):
        if slippage_model not in ("fixed", "sqrt"):
            raise ValueError("slippage_model 只能是 fixed 或 sqrt")
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.slippage = slippage
        self.stamp_duty_rate = stamp_duty_rate
        self.transfer_fee_rate = transfer_fee_rate
        self.lot_size = lot_size
        self.slippage_model = slippage_model
        self.impact_coeff = max(0.0, impact_coeff)
        self.max_impact = max(0.0, max_impact)

    # ---- 公开接口 ----

    def execute_buy(self, price: float, shares: int,
                    daily_amount: float = None) -> Fill:
        """
        执行买入订单。

        Args:
            price:        计划买入价（通常为次日开盘价）
            shares:       买入股数
            daily_amount: 当日成交额（平方根冲击模型用，缺失时按 0 参与率）

        Returns:
            Fill 成交信息
        """
        shares = self._round_shares(shares)
        if shares <= 0:
            raise ValueError(f"买入股数必须 > 0: {shares}")

        part = self.participation(shares, price, daily_amount)
        impact = self.impact_rate(part)
        exec_price = self._slippage_price(price, OrderSide.BUY, part)
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
            participation=part,
            impact_rate=impact,
        )

    def execute_sell(self, price: float, shares: int,
                     daily_amount: float = None) -> Fill:
        """
        执行卖出订单。

        Args:
            price:        计划卖出价
            shares:       卖出股数
            daily_amount: 当日成交额（平方根冲击模型用）

        Returns:
            Fill 成交信息
        """
        shares = self._round_shares(shares)
        if shares <= 0:
            raise ValueError(f"卖出股数必须 > 0: {shares}")

        part = self.participation(shares, price, daily_amount)
        impact = self.impact_rate(part)
        exec_price = self._slippage_price(price, OrderSide.SELL, part)
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
            participation=part,
            impact_rate=impact,
        )

    def calc_shares(self, cash: float, price: float,
                    fraction: float = 1.0, daily_amount: float = None) -> int:
        """
        根据可用资金计算可买入的股数。

        Args:
            cash:         可用资金
            price:        买入价
            fraction:     资金使用比例（默认 1.0 = 全仓）
            daily_amount: 当日成交额（平方根模型下用于估计冲击成本）

        Returns:
            可买入股数（取整到手的倍数），确保不超资金
        """
        budget = cash * fraction
        effective_price = price * (1 + self.slippage)

        # 参与率取决于股数，股数又取决于含冲击的成交价，迭代几次收敛。
        # 冲击随股数单调上升, 每轮都会把股数往下调, 不会来回震荡。
        if self.slippage_model == "sqrt" and daily_amount:
            for _ in range(6):
                raw_shares = int(budget / effective_price)
                guess = self._round_shares(raw_shares)
                part = self.participation(guess, price, daily_amount)
                new_price = price * (1 + self.slippage
                                     + self.impact_rate(part))
                if abs(new_price - effective_price) < 1e-12:
                    break
                effective_price = new_price

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

    # ---- 冲击成本 (Phase 16) ----

    @staticmethod
    def participation(shares: int, price: float,
                      daily_amount: float = None) -> float:
        """参与率 = 本笔成交额 / 当日成交额，缺失或不合法时返回 0。"""
        if daily_amount is None:
            return 0.0
        try:
            amt = float(daily_amount)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(amt) or amt <= 0:
            return 0.0
        return max(0.0, shares * price / amt)

    def impact_rate(self, participation: float) -> float:
        """平方根冲击成本（比例）。fixed 模型恒为 0。"""
        if self.slippage_model != "sqrt" or participation <= 0:
            return 0.0
        return min(self.impact_coeff * math.sqrt(participation),
                   self.max_impact)

    # ---- 内部方法 ----

    def _slippage_price(self, price: float, side: OrderSide,
                        participation: float = 0.0) -> float:
        """计算滑点后的成交价（固定滑点 + 可选冲击成本）"""
        slip = self.slippage + self.impact_rate(participation)
        if side == OrderSide.BUY:
            return price * (1 + slip)
        else:
            return price * (1 - slip)

    def _calc_commission(self, price: float, shares: int) -> float:
        """计算佣金（含最低 5 元限制）"""
        raw = price * shares * self.commission_rate
        return max(raw, self.min_commission)

    def _round_shares(self, shares: int) -> int:
        """取整到手的倍数（向下）"""
        return (shares // self.lot_size) * self.lot_size
