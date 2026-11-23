"""
IC 序列的显著性推断：Newey-West 与 block bootstrap。

上一阶段的 ic_sensitivity 用 Newey-West 修正重叠收益带来的自相关。
Newey-West 是一个解析近似：用 Bartlett 核估计长期方差，lag 需要自己
选。这一阶段加一个不依赖解析公式的对照 —— block bootstrap。

思路很直接：IC 序列有正自相关，单点重采样会破坏这种结构，所以按
"块"重采样。每块内部的顺序和相关性保留，块与块之间当作可以互换。
常用的两种：

    circular     把序列首尾接成环，从任意位置取长度为 L 的块
    stationary   块长服从几何分布，平均块长 L（Politis-Romano）

块长跟评估期限有关：前向收益重叠 H-1 天，IC 序列的自相关大致在 H 天
以内，块长取 H 就能覆盖。bootstrap 给出的是均值 IC 的抽样分布，标准误
和置信区间都从这组重采样均值里算，不需要假设 t 分布。

注意：bootstrap 修的是"时间维度的自相关"，修不了样本量小和横截面
太窄的问题。25 只股票的池子上，两种方法都只能说"不显著"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from src.analysis.ic_sensitivity import newey_west_tstat

__all__ = [
    "BootstrapStats",
    "InferenceComparison",
    "circular_block_resample",
    "stationary_block_resample",
    "block_bootstrap_mean",
    "compare_inference",
    "format_inference_report",
]


# ===================================================================
# 重采样
# ===================================================================

def _clean(values: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(values) if not isinstance(values, np.ndarray)
                   else values, dtype=float)
    return x[np.isfinite(x)]


def _circular_indices(n: int, block_size: int, n_boot: int,
                      rng: np.random.Generator) -> np.ndarray:
    """环形块重采样的下标矩阵，形状 (n_boot, n)。"""
    blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_boot, blocks))
    offsets = np.arange(block_size)
    idx = (starts[:, :, None] + offsets[None, None, :]) % n
    return idx.reshape(n_boot, blocks * block_size)[:, :n]


def _stationary_indices(n: int, mean_block: float, n_boot: int,
                        rng: np.random.Generator) -> np.ndarray:
    """
    平稳 bootstrap（Politis-Romano）的下标矩阵。

    每一步以 p = 1/L 的概率跳到新的随机起点，否则沿序列往后走一格。
    块长因此服从均值为 L 的几何分布。用矩阵一次生成所有样本的路径，
    只在外层循环 n 步，比逐个样本循环快得多。
    """
    p = 1.0 / max(1.0, float(mean_block))
    pos = np.empty((n_boot, n), dtype=np.int64)
    pos[:, 0] = rng.integers(0, n, size=n_boot)

    for t in range(1, n):
        jump = rng.random(n_boot) < p
        nxt = (pos[:, t - 1] + 1) % n
        pos[:, t] = np.where(jump, rng.integers(0, n, size=n_boot), nxt)
    return pos


def circular_block_resample(values: Sequence[float], block_size: int,
                            n_boot: int = 2000,
                            seed: int = 0) -> np.ndarray:
    """返回 n_boot 个环形块重采样均值。"""
    x = _clean(values)
    if x.size < 2 or block_size <= 0 or n_boot <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    idx = _circular_indices(x.size, min(int(block_size), x.size), int(n_boot), rng)
    return x[idx].mean(axis=1)


def stationary_block_resample(values: Sequence[float], block_size: float,
                              n_boot: int = 2000,
                              seed: int = 0) -> np.ndarray:
    """返回 n_boot 个平稳 bootstrap 重采样均值。"""
    x = _clean(values)
    if x.size < 2 or block_size <= 0 or n_boot <= 0:
        return np.array([], dtype=float)
    rng = np.random.default_rng(seed)
    idx = _stationary_indices(x.size, float(block_size), int(n_boot), rng)
    return x[idx].mean(axis=1)


# ===================================================================
# 统计量
# ===================================================================

@dataclass
class BootstrapStats:
    """一组 bootstrap 重采样均值的汇总"""

    mean: float
    se: float
    t_stat: float          # mean / se
    p_two_sided: float     # 重采样均值落在 0 另一侧的比例（双侧）
    ci_low: float
    ci_high: float
    n_boot: int
    block_size: float
    method: str

    def __repr__(self) -> str:
        return (f"BootstrapStats(mean={self.mean:.4f}, se={self.se:.4f}, "
                f"t={self.t_stat:.2f}, CI=[{self.ci_low:.4f}, "
                f"{self.ci_high:.4f}], L={self.block_size}, {self.method})")


def _rule_of_thumb_block(n: int) -> int:
    """没有先验时的块长：n^(1/3)，够覆盖短期相关，又不至于只剩几块。"""
    return max(1, int(round(n ** (1.0 / 3.0))))


def block_bootstrap_mean(
    values: Sequence[float],
    *,
    block_size: Optional[float] = None,
    n_boot: int = 2000,
    seed: int = 0,
    method: str = "circular",
    ci: float = 0.95,
) -> BootstrapStats:
    """
    对均值做块 bootstrap。

    Parameters
    ----------
    block_size : 块长；None 时用 n^(1/3)。做前向收益重叠修正时
                 调用方一般直接传 horizon。
    method     : "circular" | "stationary"
    ci         : 百分位置信区间水平
    """
    if method not in ("circular", "stationary"):
        raise ValueError("method must be 'circular' or 'stationary'")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")

    x = _clean(values)
    n = x.size
    if n < 2 or n_boot <= 0:
        return BootstrapStats(float("nan"), float("nan"), 0.0, 1.0,
                              float("nan"), float("nan"), 0, 0.0, method)

    L = float(block_size) if block_size is not None else float(_rule_of_thumb_block(n))
    L = max(1.0, min(L, float(n)))

    if method == "circular":
        means = circular_block_resample(x, int(round(L)), n_boot=n_boot, seed=seed)
    else:
        means = stationary_block_resample(x, L, n_boot=n_boot, seed=seed)

    mean = float(x.mean())
    se = float(means.std(ddof=1))
    t_stat = mean / se if se > 0 else 0.0

    # 双侧 p 值：重采样均值里跨到 0 另一侧的比例（+1 平滑）
    if mean > 0:
        tail_count = int(np.sum(means <= 0.0))
    elif mean < 0:
        tail_count = int(np.sum(means >= 0.0))
    else:
        tail_count = int(len(means) // 2)     # 正好 0，双侧各占一半
    p = min(1.0, 2.0 * (tail_count + 1.0) / (len(means) + 1.0))

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(means, [100.0 * alpha, 100.0 * (1.0 - alpha)])

    return BootstrapStats(
        mean=mean,
        se=se,
        t_stat=t_stat,
        p_two_sided=p,
        ci_low=float(lo),
        ci_high=float(hi),
        n_boot=int(n_boot),
        block_size=L,
        method=method,
    )


@dataclass
class InferenceComparison:
    """同一条 IC 序列上三种推断方式的结果"""

    mean: float
    n: int
    t_naive: float         # mean / (std / sqrt(n))
    t_nw: float            # Newey-West，lag 由调用方给
    lags: int
    boot: BootstrapStats

    @property
    def t_boot(self) -> float:
        return self.boot.t_stat

    @property
    def boot_se(self) -> float:
        return self.boot.se

    def __repr__(self) -> str:
        return (f"InferenceComparison(mean={self.mean:.4f}, n={self.n}, "
                f"naive={self.t_naive:.2f}, nw={self.t_nw:.2f}, "
                f"boot={self.t_boot:.2f})")


def compare_inference(
    values: Sequence[float],
    *,
    lags: int,
    block_size: Optional[float] = None,
    n_boot: int = 2000,
    seed: int = 0,
    method: str = "circular",
    ci: float = 0.95,
) -> InferenceComparison:
    """
    对一段 IC 序列同时算普通 t、Newey-West t 和 block bootstrap。

    lags       : Newey-West 的滞后阶数，重叠收益一般取 horizon - 1
    block_size : bootstrap 块长，None 时取 lags + 1（即 horizon）
    """
    x = _clean(values)
    n = x.size
    mean = float(x.mean()) if n else float("nan")

    if n >= 2:
        std = float(x.std(ddof=1))
        t_naive = mean / (std / np.sqrt(n)) if std > 0 else 0.0
        t_nw = newey_west_tstat(x, lags)
    else:
        t_naive = 0.0
        t_nw = 0.0

    L = block_size if block_size is not None else float(max(1, int(lags) + 1))
    boot = block_bootstrap_mean(
        x, block_size=L, n_boot=n_boot, seed=seed, method=method, ci=ci,
    )

    return InferenceComparison(
        mean=mean,
        n=int(n),
        t_naive=float(t_naive),
        t_nw=float(t_nw),
        lags=int(lags),
        boot=boot,
    )


# ===================================================================
# 文本报告
# ===================================================================

def format_inference_report(
    results: Dict[Tuple[str, int], InferenceComparison],
    factors: Sequence[str],
    horizons: Sequence[int],
    *,
    title: str = "IC 显著性：Newey-West vs block bootstrap",
) -> str:
    """把 {factor: {horizon: 对比结果}} 摊成一张表。"""
    lines = ["=" * 78, f"  {title}", "=" * 78, ""]
    lines.append(
        f"  {'因子':<14}{'h':>3}{'mean_IC':>10}{'t_naive':>9}"
        f"{'t_NW':>8}{'t_boot':>9}{'boot_SE':>9}{'95% CI':>22}"
    )
    lines.append("  " + "-" * 74)

    for name in factors:
        for h in horizons:
            r = results.get((name, int(h)))
            if r is None:
                continue
            ci = f"[{r.boot.ci_low:+.4f},{r.boot.ci_high:+.4f}]"
            lines.append(
                f"  {name:<14}{h:>3}{r.mean:>10.4f}{r.t_naive:>9.2f}"
                f"{r.t_nw:>8.2f}{r.t_boot:>9.2f}{r.boot_se:>9.4f}{ci:>22}"
            )

    lines.append("")
    lines.append("  读法:")
    lines.append("  - t_naive 把重叠样本当独立样本，期限越长越不可信。")
    lines.append("  - t_NW 和 t_boot 都修了时间自相关；两者差距大时说明")
    lines.append("    lag/块长选得不合适，别只挑一个顺眼的看。")
    lines.append("  - 95% CI 跨过 0，基本可以当作这条因子在这组数据上")
    lines.append("    没有可辨识的方向。")
    lines.append("=" * 78)
    return "\n".join(lines)
