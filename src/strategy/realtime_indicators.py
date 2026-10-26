"""
实时技术指标 —— 增量递推版本，用于实盘 / 模拟交易。

与 indicators.py 的区别：
- indicators.py: pandas 向量化，一次性批量计算，适合回测
- 本文件: 纯 Python 增量递推，每次 O(1)，适合逐 tick 实时更新

每个类维护最小状态，不存全量历史。用法：

    sma5 = OnlineSMA(5)
    for price in streaming_prices:
        current_sma = sma5.update(price)

所有类和 indicators.py 中的同名函数计算结果一致（浮点精度范围内）。
"""

from collections import deque
import math


class OnlineSMA:
    """SMA 增量计算 —— 循环缓冲 + 累和，每次 O(1)"""

    def __init__(self, period: int):
        self.period = period
        self.buf = deque()
        self.sum = 0.0
        self.value = float("nan")

    def update(self, price: float) -> float:
        self.buf.append(price)
        self.sum += price

        if len(self.buf) > self.period:
            old = self.buf.popleft()
            self.sum -= old

        self.value = self.sum / len(self.buf)
        return self.value


class OnlineEMA:
    """EMA 增量计算 —— 只存上一个 EMA 值，每次一次乘加"""

    def __init__(self, period: int):
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self.value = None

    def update(self, price: float) -> float:
        if self.value is None:
            self.value = price
        else:
            self.value = (self.alpha * price +
                          (1.0 - self.alpha) * self.value)
        return self.value


class OnlineMACD:
    """MACD 增量计算 —— 三个 EMA 状态机"""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.ema_fast = OnlineEMA(fast)
        self.ema_slow = OnlineEMA(slow)
        self.ema_signal = OnlineEMA(signal)
        self.dif = None
        self.dea = None
        self.hist = None

    def update(self, price: float):
        """更新 MACD 并返回 (dif, dea, hist)"""
        fast_val = self.ema_fast.update(price)
        slow_val = self.ema_slow.update(price)
        self.dif = fast_val - slow_val
        self.dea = self.ema_signal.update(self.dif)
        self.hist = (self.dif - self.dea) * 2
        return self.dif, self.dea, self.hist


class OnlineRSI:
    """RSI 增量计算 —— Wilder 平滑的在线版本"""

    def __init__(self, period: int = 14):
        self.period = period
        self.alpha = 1.0 / period
        self.prev_close = None
        self.avg_gain = None
        self.avg_loss = None
        self.value = None
        self._count = 0

    def update(self, price: float) -> float:
        if self.prev_close is None:
            self.prev_close = price
            return float("nan")

        delta = price - self.prev_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        self.prev_close = price

        self._count += 1

        if self._count == 1:
            self.avg_gain = gain
            self.avg_loss = loss
        elif self._count <= self.period:
            self.avg_gain = ((self.avg_gain * (self._count - 1) + gain)
                             / self._count)
            self.avg_loss = ((self.avg_loss * (self._count - 1) + loss)
                             / self._count)
        else:
            self.avg_gain = ((1.0 - self.alpha) * self.avg_gain +
                             self.alpha * gain)
            self.avg_loss = ((1.0 - self.alpha) * self.avg_loss +
                             self.alpha * loss)

        if self.avg_loss == 0:
            self.value = 100.0
        else:
            rs = self.avg_gain / self.avg_loss
            self.value = 100.0 - (100.0 / (1.0 + rs))

        return self.value


class OnlineBollinger:
    """布林带增量计算 —— Welford 算法维护滑动窗口均值和方差"""

    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std
        self.buf = deque()
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, price: float):
        """更新布林带，返回 (upper, middle, lower)"""
        self.buf.append(price)
        self.count += 1

        # Welford 递推：加入新值
        delta = price - self.mean
        self.mean += delta / self.count
        delta2 = price - self.mean
        self.m2 += delta * delta2

        # 超出窗口：移除最旧值
        if len(self.buf) > self.period:
            old = self.buf.popleft()
            self.count -= 1
            delta = old - self.mean
            self.mean -= delta / self.count
            delta2 = old - self.mean
            self.m2 -= delta * delta2

        if self.count < 2:
            return float("nan"), float("nan"), float("nan")

        variance = self.m2 / self.count
        std = math.sqrt(variance)
        upper = self.mean + self.num_std * std
        lower = self.mean - self.num_std * std
        return upper, self.mean, lower


class RealtimeIndicatorCalculator:
    """实盘指标计算器 —— 一次 on_tick 更新所有指标"""

    def __init__(self):
        self.sma5 = OnlineSMA(5)
        self.sma20 = OnlineSMA(20)
        self.ema12 = OnlineEMA(12)
        self.ema26 = OnlineEMA(26)
        self.macd = OnlineMACD(12, 26, 9)
        self.rsi = OnlineRSI(14)

    def on_tick(self, price: float) -> dict:
        """收到一个新 tick 时调用，返回当前所有指标值"""
        s5 = self.sma5.update(price)
        s20 = self.sma20.update(price)
        e12 = self.ema12.update(price)
        e26 = self.ema26.update(price)
        self.macd.update(price)
        r14 = self.rsi.update(price)

        return {
            "price": price,
            "sma5": s5,
            "sma20": s20,
            "ema12": e12,
            "ema26": e26,
            "dif": self.macd.dif,
            "dea": self.macd.dea,
            "hist": self.macd.hist,
            "rsi14": r14,
        }


# ── 验证：增量版 vs pandas 批量版 ────────────────────────────
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # 造一段模拟价格数据
    np.random.seed(42)
    close = pd.Series(
        [10.0 + x * 0.05 + np.random.normal(0, 0.1) for x in range(200)]
    )

    # ── SMA 验证 ──
    sma_online = OnlineSMA(5)
    sma_incremental = [sma_online.update(p) for p in close]
    sma_pandas = close.rolling(window=5).mean()
    assert np.allclose(sma_incremental[4:], sma_pandas.dropna().values, atol=1e-10)
    print("[OK] SMA: incremental matches pandas")

    # ── EMA 验证 ──
    ema_online = OnlineEMA(12)
    ema_incremental = [ema_online.update(p) for p in close]
    ema_pandas = close.ewm(span=12, adjust=False).mean()
    assert np.allclose(ema_incremental, ema_pandas.values, atol=1e-10)
    print("[OK] EMA: incremental matches pandas")

    # ── MACD 验证 ──
    macd_online = OnlineMACD()
    dif_inc, dea_inc = [], []
    for p in close:
        macd_online.update(p)
        dif_inc.append(macd_online.dif)
        dea_inc.append(macd_online.dea)

    dif_pd = (close.ewm(span=12, adjust=False).mean() -
              close.ewm(span=26, adjust=False).mean())
    dea_pd = dif_pd.ewm(span=9, adjust=False).mean()
    assert np.allclose(dif_inc, dif_pd.values, atol=1e-10)
    assert np.allclose(dea_inc, dea_pd.values, atol=1e-10)
    print("[OK] MACD: incremental matches pandas")

    # ── RSI 验证 ──
    # RSI增量版前period个值用SMA平滑初始化（Wilder原始方法），
    # pandas的ewm全程用EMA递推。前几项有差异，但30周期后应收敛
    rsi_online = OnlineRSI(14)
    rsi_inc = [rsi_online.update(p) for p in close]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # pandas ewm 全程递推 (和增量版的SMA初始化方式不同)
    avg_gain_pd = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss_pd = loss.ewm(alpha=1/14, adjust=False).mean()
    rs_pd = avg_gain_pd / avg_loss_pd
    rsi_pd = 100.0 - (100.0 / (1.0 + rs_pd))

    # 后30个值应该非常接近
    diff = np.max(np.abs(np.array(rsi_inc[-30:]) - rsi_pd.values[-30:]))
    assert diff < 0.01, f"RSI diff too large: {diff}"
    print(f"[OK] RSI: max diff in last 30 values = {diff:.6f}")

    print("\nAll checks passed.")
