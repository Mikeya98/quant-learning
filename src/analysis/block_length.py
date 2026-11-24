"""
Politis-White 自动块长选择。

上一阶段的 block bootstrap 里，块长是手工指定的：做前向收益重叠修正
时取 L = horizon。这个选择有一个隐含假设 —— IC 序列的自相关恰好持续
到 H 天。实际上自相关有多长，要看数据本身。

Politis & White (2004) 给了一个不靠人拍脑袋的办法。思路是：

    1. 算样本自相关，做一次"隐含假设检验"，找出自相关衰减到噪声
       水平的位置 m_hat；
    2. 用 m_hat 定一个截断 M = 2*m_hat；
    3. 在 M 以内用平顶核估计 Σ|k|γ(k)，代入最优块长的解析式

        b_opt = ( 2 * G^2 / (c_i * sigma^4) * n )^(1/3)

    其中 G = Σ_{k=-M..M} h(k/M)|k|γ(k)，h(x) = min(1, 2(1-|x|))，
    c_i 对平稳 bootstrap 取 2、对环形 bootstrap 取 4/3
    （Patton, Politis & White 2009 的修正）。

指数 1/3 来自均值方差估计的偏差-方差权衡：块太短丢相关性，块太长
有效样本数变少，两头的误差在 n^(1/3) 处平衡。所以 n 越大，块长增长
得越慢 —— 1400 个交易日的数据，最优块长通常只有个位数。

依赖只有 numpy，不依赖 ic_inference，方便单独测试。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

__all__ = [
    "BlockLengthResult",
    "sample_acf",
    "politi_white_block_length",
]


@dataclass
class BlockLengthResult:
    """一次 Politis-White 估计的完整中间量，便于写报告和排查。"""

    n: int
    m_hat: int              # 自相关衰减到噪声水平的滞后
    max_lag: int            # 平顶核的截断 M = 2*m_hat
    b_max: int              # 块长上限 ceil(min(3√n, n/3))
    b_sb: float             # 平稳 bootstrap 最优平均块长
    b_cb: float             # 环形 bootstrap 最优固定块长
    sigma2: float           # 样本方差（除以 n）
    g_hat: float            # Σ h(k/M)|k|γ(k)（含对称项）
    tolerance: float        # 隐含假设检验的阈值 c√(log10 n / n)

    @property
    def block_size(self) -> float:
        """默认用平稳 bootstrap 的块长（本项目的默认抽样方式）。"""
        return self.b_sb

    def __repr__(self) -> str:
        return (f"BlockLengthResult(n={self.n}, m_hat={self.m_hat}, "
                f"M={self.max_lag}, b_sb={self.b_sb:.1f}, "
                f"b_cb={self.b_cb:.1f})")


def _clean(values: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(values) if not isinstance(values, np.ndarray)
                   else values, dtype=float)
    return x[np.isfinite(x)]


def sample_acf(values: Iterable[float], max_lag: int) -> np.ndarray:
    """
    样本自相关 ρ(0)..ρ(max_lag)。

    自协方差除以 n（不是 n-k），和 Newey-West / Politis-White 的
    Bartlett 型估计口径一致；这样 ρ(0) 恰好是 1。
    """
    x = _clean(values)
    n = x.size
    if n == 0:
        return np.zeros(0)
    max_lag = max(0, min(int(max_lag), n - 1))
    e = x - x.mean()
    denom = float(e @ e) / n
    if denom <= 0.0:
        return np.zeros(max_lag + 1)
    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for k in range(1, max_lag + 1):
        acf[k] = float(e[k:] @ e[:-k]) / n / denom
    return acf


def _implicit_m_hat(acf: np.ndarray, n: int, k_n: int,
                    tolerance: float, m_max: int) -> int:
    """
    找最小的 m，使 m+1 之后连续 k_n 个自相关都不超过噪声阈值。

    找不到这样的 run 时退回 m_max —— 说明序列自相关很长，块长上限
    会把它压住，报告里也能从 m_hat == m_max 看出来。
    """
    for m in range(0, m_max + 1):
        lo = m + 1
        if lo > m_max:
            return m
        hi = min(m_max, m + k_n)
        if np.all(np.abs(acf[lo:hi + 1]) <= tolerance):
            return m
    return m_max


def politi_white_block_length(
    values: Iterable[float],
    *,
    method: str = "stationary",
    c: float = 2.0,
    m_hat: Optional[int] = None,
) -> BlockLengthResult:
    """
    估计序列均值的 Politis-White 最优块长。

    Parameters
    ----------
    method : "stationary" | "circular"
        决定返回哪个常数对应的块长：平稳 bootstrap 的 c_i = 2，
        环形 bootstrap 的 c_i = 4/3，前者比后者短约 12.6%。
    c : float
        隐含假设检验的阈值系数，原文推荐 2。
    m_hat : int, optional
        手动指定衰减滞后，跳过自动检验（调试/敏感性分析用）。

    样本太短、方差为 0 或者自相关近乎为 0 时，b_opt 会被夹到
    [1, b_max] 的 1 端 —— 块长 1 就是普通的单点 bootstrap。
    """
    if method not in ("stationary", "circular"):
        raise ValueError("method must be 'stationary' or 'circular'")
    if c <= 0.0:
        raise ValueError("c must be positive")

    x = _clean(values)
    n = x.size
    if n < 16:
        return BlockLengthResult(n, 0, 0, 1, 1.0, 1.0, 0.0, 0.0, float("nan"))

    sigma2 = float(x.var(ddof=0))
    if sigma2 <= 0.0:
        return BlockLengthResult(n, 0, 0, 1, 1.0, 1.0, 0.0, 0.0, 0.0)

    log10n = np.log10(n)
    k_n = max(5, int(np.ceil(log10n)))
    m_max = int(np.ceil(np.sqrt(n))) + k_n
    tolerance = float(c * np.sqrt(log10n / n))

    # 平顶核最多用到 M = 2*m_hat，所以自相关要算到 2*m_max。
    acf = sample_acf(x, 2 * m_max)

    if m_hat is None:
        m_hat = _implicit_m_hat(acf, n, k_n, tolerance, m_max)
    m_hat = max(0, min(int(m_hat), m_max))
    M = 2 * m_hat

    b_max = int(np.ceil(min(3.0 * np.sqrt(n), n / 3.0)))

    if M <= 0:
        return BlockLengthResult(n, m_hat, M, b_max, 1.0, 1.0,
                                 sigma2, 0.0, tolerance)

    gamma = acf * sigma2
    j = np.arange(1, M + 1, dtype=float)
    h = np.minimum(1.0, 2.0 * (1.0 - j / M))       # 平顶（梯形）核
    g_hat = float(2.0 * np.sum(h * j * gamma[1:M + 1]))

    # 白噪声时 G ≈ 0，b_opt 为 0，夹到 1 就是不过度分块。
    if g_hat <= 0.0:
        b_sb = b_cb = 1.0
    else:
        base = 2.0 * g_hat * g_hat / (sigma2 * sigma2) * n
        b_sb = float(base / 2.0) ** (1.0 / 3.0)       # c_i = 2
        b_cb = float(base / (4.0 / 3.0)) ** (1.0 / 3.0)

    b_sb = float(np.clip(b_sb, 1.0, b_max))
    b_cb = float(np.clip(b_cb, 1.0, b_max))

    return BlockLengthResult(
        n=n,
        m_hat=int(m_hat),
        max_lag=int(M),
        b_max=int(b_max),
        b_sb=b_sb,
        b_cb=b_cb,
        sigma2=sigma2,
        g_hat=g_hat,
        tolerance=tolerance,
    )
