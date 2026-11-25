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

Phase 24 在区间上又加了一层：百分位区间假设重采样分布无偏、对称，
有限样本里不成立；BCa 用 z0 修中位数偏差、用刀切加速度修偏度。
但对 AR(1) 的覆盖率模拟显示，块长不足造成的 SE 低估它修不了 ——
区间修正不能替代块长选择。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from src.analysis.block_length import politi_white_block_length
from src.analysis.ic_sensitivity import newey_west_tstat

__all__ = [
    "BootstrapStats",
    "InferenceComparison",
    "CoverageResult",
    "circular_block_resample",
    "stationary_block_resample",
    "block_bootstrap_mean",
    "bootstrap_se_curve",
    "bca_interval",
    "normal_cdf",
    "normal_ppf",
    "coverage_simulation",
    "compare_inference",
    "format_inference_report",
    "format_coverage_report",
]


# ===================================================================
# 重采样
# ===================================================================

def _clean(values: Iterable[float]) -> np.ndarray:
    x = np.asarray(list(values) if not isinstance(values, np.ndarray)
                   else values, dtype=float)
    return x[np.isfinite(x)]


# ===================================================================
# 正态分布辅助函数（BCa 需要）
# ===================================================================

def normal_cdf(z: float) -> float:
    """标准正态分布的累积分布函数。"""
    return 0.5 * math.erfc(-float(z) / math.sqrt(2.0))


def normal_ppf(p: float) -> float:
    """
    标准正态分布的分位数（Acklam 有理逼近）。

    只要 p 在 (0,1) 开区间内就够了，精度约 1e-9。p<=0 / p>=1 返回
    ±inf，调用方自己决定怎么夹。
    """
    p = float(p)
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
             + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                             + b[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
              + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

    # 一步 Halley 修正，把误差压到机器精度
    e = normal_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


# ===================================================================
# BCa 区间
# ===================================================================

def bca_interval(
    means: Sequence[float],
    values: Sequence[float],
    *,
    ci: float = 0.95,
) -> Tuple[float, float, float, float]:
    """
    均值的偏差校正-加速（BCa）置信区间。

    百分位区间只做了"重采样分布"这一件事，有两个已知偏差：
      - 中位数偏差：bootstrap 分布的中心不一定落在样本统计量上，
        用 z0 = Φ^-1(P(θ* < θ̂)) 修正；
      - 偏度/方差随参数变化：用刀切法估一个加速度 a，把区间两端
        按 a 拉伸或压缩。

    返回 (lo, hi, z0, a)。样本太短或重采样次数太少时退化成百分位区间，
    此时 z0 = 0、a = 0。
    """
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")

    boot = np.asarray(means, dtype=float)
    boot = boot[np.isfinite(boot)]
    x = _clean(values)
    n = x.size

    if boot.size < 10 or n < 3:
        alpha = (1.0 - ci) / 2.0
        if boot.size == 0:
            return float("nan"), float("nan"), 0.0, 0.0
        lo, hi = np.percentile(boot, [100.0 * alpha, 100.0 * (1.0 - alpha)])
        return float(lo), float(hi), 0.0, 0.0

    theta = float(x.mean())

    # 刀切法：去掉第 i 个观测后的均值。对均值来说有闭式解，
    # 不用真的重算 n 次。
    jack = (n * theta - x) / (n - 1.0)
    d = float(jack.mean()) - jack
    denom = 6.0 * float(np.sum(d * d)) ** 1.5
    accel = float(np.sum(d * d * d) / denom) if denom > 0.0 else 0.0
    if not math.isfinite(accel):
        accel = 0.0

    # 重采样分布里落在 θ̂ 以下的比例。用 0.5/B 夹一下，避免 z0 = ±inf。
    prop = float(np.mean(boot < theta))
    prop = min(max(prop, 0.5 / boot.size), 1.0 - 0.5 / boot.size)
    z0 = normal_ppf(prop)

    alpha = (1.0 - ci) / 2.0

    def _adjust(z_alpha: float) -> float:
        num = z0 + z_alpha
        den = 1.0 - accel * num
        if den <= 0.0:
            return float("nan")
        return normal_cdf(z0 + num / den)

    lo_p = _adjust(normal_ppf(alpha))
    hi_p = _adjust(normal_ppf(1.0 - alpha))

    if not (math.isfinite(lo_p) and math.isfinite(hi_p)) or lo_p >= hi_p:
        lo, hi = np.percentile(boot, [100.0 * alpha, 100.0 * (1.0 - alpha)])
        return float(lo), float(hi), float(z0), float(accel)

    lo_p = min(max(lo_p, 0.0), 1.0)
    hi_p = min(max(hi_p, 0.0), 1.0)
    lo, hi = np.percentile(boot, [100.0 * lo_p, 100.0 * hi_p])
    return float(lo), float(hi), float(z0), float(accel)


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
    block_source: str = "fixed"     # fixed / horizon / pw-sb / pw-cb

    # v0.24: BCa（偏差校正-加速）区间，和百分位区间一起给出
    ci_low_bca: float = float("nan")
    ci_high_bca: float = float("nan")
    bca_z0: float = 0.0
    bca_accel: float = 0.0

    def __repr__(self) -> str:
        return (f"BootstrapStats(mean={self.mean:.4f}, se={self.se:.4f}, "
                f"t={self.t_stat:.2f}, CI=[{self.ci_low:.4f}, "
                f"{self.ci_high:.4f}], BCa=[{self.ci_low_bca:.4f}, "
                f"{self.ci_high_bca:.4f}], L={self.block_size}, "
                f"{self.method})")


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
    block_source: str = "fixed",
) -> BootstrapStats:
    """
    对均值做块 bootstrap。

    Parameters
    ----------
    block_size : 块长；None 时用 n^(1/3)。做前向收益重叠修正时
                 调用方一般直接传 horizon。
    method     : "circular" | "stationary"
    ci         : 百分位置信区间水平
    block_source : 块长来源标记，只用于报告（fixed / horizon / pw-sb / pw-cb）
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
    bca_lo, bca_hi, z0, accel = bca_interval(means, x, ci=ci)

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
        block_source=block_source,
        ci_low_bca=bca_lo,
        ci_high_bca=bca_hi,
        bca_z0=z0,
        bca_accel=accel,
    )


def bootstrap_se_curve(
    values: Sequence[float],
    block_sizes: Sequence[float] = (1, 2, 5, 10, 20, 40),
    *,
    n_boot: int = 2000,
    seed: int = 0,
    method: str = "circular",
    ci: float = 0.95,
) -> list:
    """
    扫一组块长，返回每个块长下的 SE / t / 置信区间。

    用来回答"块长到底要多长"。正自相关时，块长从 1 加到自相关的
    尺度上，SE 会明显上升，之后趋于平台 —— 平台起点就是自相关被
    覆盖的位置。序列近似白噪声或 lag-1 为负时，SE 对块长不敏感，
    甚至随 L 缓慢下降（块越长、有效样本越少）。
    """
    rows = []
    for L in block_sizes:
        st = block_bootstrap_mean(
            values, block_size=L, n_boot=n_boot, seed=seed,
            method=method, ci=ci,
        )
        rows.append({
            "block_size": float(L),
            "se": st.se,
            "t": st.t_stat,
            "ci_low": st.ci_low,
            "ci_high": st.ci_high,
        })
    return rows


@dataclass
class InferenceComparison:
    """同一条 IC 序列上三种推断方式的结果"""

    mean: float
    n: int
    t_naive: float         # mean / (std / sqrt(n))
    t_nw: float            # Newey-West，lag 由调用方给
    lags: int
    boot: BootstrapStats
    block_source: str = "horizon"

    @property
    def t_boot(self) -> float:
        return self.boot.t_stat

    @property
    def block_length(self) -> float:
        return self.boot.block_size

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
    block_method: str = "horizon",
    n_boot: int = 2000,
    seed: int = 0,
    method: str = "circular",
    ci: float = 0.95,
) -> InferenceComparison:
    """
    对一段 IC 序列同时算普通 t、Newey-West t 和 block bootstrap。

    lags         : Newey-West 的滞后阶数，重叠收益一般取 horizon - 1
    block_size   : bootstrap 块长，给定时优先；None 时看 block_method
    block_method : "horizon"（取 lags+1，即评估期限）
                   "pw-sb" / "pw-cb"（Politis-White 自适应块长）
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

    if block_size is not None:
        L = float(block_size)
        source = "fixed"
    elif block_method in ("pw-sb", "pw-cb"):
        pw = politi_white_block_length(
            x, method="stationary" if block_method == "pw-sb" else "circular",
        )
        L = pw.b_sb if block_method == "pw-sb" else pw.b_cb
        source = block_method
    elif block_method == "horizon":
        L = float(max(1, int(lags) + 1))
        source = "horizon"
    else:
        raise ValueError("block_method must be 'horizon', 'pw-sb' or 'pw-cb'")

    boot = block_bootstrap_mean(
        x, block_size=L, n_boot=n_boot, seed=seed, method=method, ci=ci,
        block_source=source,
    )

    return InferenceComparison(
        mean=mean,
        n=int(n),
        t_naive=float(t_naive),
        t_nw=float(t_nw),
        lags=int(lags),
        boot=boot,
        block_source=source,
    )


# ===================================================================
# 覆盖率模拟：块长选得对不对，用已知真值的合成数据检验
# ===================================================================

@dataclass
class CoverageResult:
    """一次覆盖率模拟的结果"""

    rho: float
    n: int
    reps: int
    n_boot: int
    rows: list              # [{"label", "mean_block", "coverage"}, ...]

    def __repr__(self) -> str:
        return (f"CoverageResult(rho={self.rho}, n={self.n}, "
                f"reps={self.reps}, rows={len(self.rows)})")


def coverage_simulation(
    *,
    rho: float = 0.5,
    n: int = 800,
    reps: int = 120,
    n_boot: int = 300,
    block_sizes: Sequence[Optional[float]] = (1, 5, 10, 20, None),
    method: str = "circular",
    seed: int = 0,
    ci: float = 0.95,
) -> CoverageResult:
    """
    用已知均值为 0 的 AR(1) 序列检验置信区间的覆盖率。

    真值是 0，所以"0 落在 95% 区间里"就是一次正确覆盖。理想情况下
    覆盖率应该接近 95%。块长太短会丢掉正自相关、区间偏窄、覆盖率
    掉到 80% 以下；块长太长会让重采样均值过于平滑、覆盖率也会偏。

    block_sizes 里的 None 表示"每个样本用 Politis-White 自己选"，
    选法与 bootstrap 的抽样方式一致（circular → pw-cb）。
    另外附上普通 t 和 Newey-West（lag = 块长）的覆盖率作对照。
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    if n < 32 or reps < 10 or n_boot < 50:
        raise ValueError("n >= 32, reps >= 10, n_boot >= 50 required")

    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((reps, n))
    x = np.empty((reps, n))
    x[:, 0] = eps[:, 0] / np.sqrt(1.0 - rho * rho) if rho > 0 else eps[:, 0]
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + eps[:, t]

    # 每个样本的 PW 块长只算一次，所有用到 None 的行共用
    pw = [
        politi_white_block_length(
            x[r], method="stationary" if method == "stationary" else "circular",
        ).block_size
        for r in range(reps)
    ]

    rows = []
    for L in block_sizes:
        covered = covered_bca = 0
        lengths = []
        for r in range(reps):
            Lr = pw[r] if L is None else float(L)
            lengths.append(Lr)
            st = block_bootstrap_mean(
                x[r], block_size=Lr, n_boot=n_boot, seed=seed + r,
                method=method, ci=ci,
            )
            if st.ci_low <= 0.0 <= st.ci_high:
                covered += 1
            if st.ci_low_bca <= 0.0 <= st.ci_high_bca:
                covered_bca += 1
        rows.append({
            "label": "PW 自适应" if L is None else f"L={int(L)}",
            "mean_block": float(np.mean(lengths)),
            "coverage": covered / reps,
            "coverage_bca": covered_bca / reps,
        })

    # 解析区间对照：普通 t 和 Newey-West，lag 取 PW 对应块长
    naive_cov = nw_cov = 0
    z = 1.959963984540054
    for r in range(reps):
        seq = x[r]
        mean = float(seq.mean())
        std = float(seq.std(ddof=1))
        se = std / np.sqrt(n)
        if se > 0 and abs(mean / se) <= z:
            naive_cov += 1
        nw_t = newey_west_tstat(seq, max(0, int(round(pw[r])) - 1))
        if abs(nw_t) <= z:
            nw_cov += 1
    rows.append({"label": "普通 t", "mean_block": 1.0,
                 "coverage": naive_cov / reps, "coverage_bca": None})
    rows.append({"label": "Newey-West", "mean_block": float(np.mean(pw)),
                 "coverage": nw_cov / reps, "coverage_bca": None})

    return CoverageResult(rho=rho, n=n, reps=reps, n_boot=n_boot, rows=rows)


def format_coverage_report(result: CoverageResult,
                           nominal: float = 0.95) -> str:
    """把覆盖率模拟摊成表，顺手标出明显偏离名义水平的行。"""
    lines = ["=" * 60, "  95% 置信区间覆盖率（AR(1)，真均值 = 0）", "=" * 60, ""]
    lines.append(f"  rho = {result.rho}, n = {result.n}, "
                 f"reps = {result.reps}, n_boot = {result.n_boot}")
    lines.append(f"  名义覆盖率 = {nominal:.0%}；偏离超过 3 个百分点标 !")
    lines.append("")
    lines.append(f"  {'方法':<12}{'平均块长':>10}{'百分位':>10}{'BCa':>10}")
    lines.append("  " + "-" * 46)
    for row in result.rows:
        flag = " !" if abs(row["coverage"] - nominal) > 0.03 else ""
        bca = row.get("coverage_bca")
        bca_text = "-" if bca is None else f"{bca:>9.1%}"
        lines.append(f"  {row['label']:<12}{row['mean_block']:>10.1f}"
                     f"{row['coverage']:>9.1%}{flag:>3}{bca_text:>10}")
    lines.append("")
    lines.append("  读法: 块长 1 丢掉正自相关，百分位区间偏窄、覆盖率系统性")
    lines.append("        偏低；BCa 用偏差校正和刀切加速度把区间往需要的方向")
    lines.append("        拉，短块长下的覆盖率通常能补回来一截。")
    lines.append("=" * 60)
    return "\n".join(lines)


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
    lines = ["=" * 100, f"  {title}", "=" * 100, ""]
    lines.append(
        f"  {'因子':<14}{'h':>3}{'mean_IC':>10}{'t_naive':>9}"
        f"{'t_NW':>8}{'L':>6}{'t_boot':>9}{'boot_SE':>9}"
        f"{'CI(百分位)':>22}{'CI(BCa)':>22}"
    )
    lines.append("  " + "-" * 102)

    for name in factors:
        for h in horizons:
            r = results.get((name, int(h)))
            if r is None:
                continue
            ci = f"[{r.boot.ci_low:+.4f},{r.boot.ci_high:+.4f}]"
            bca = f"[{r.boot.ci_low_bca:+.4f},{r.boot.ci_high_bca:+.4f}]"
            lines.append(
                f"  {name:<14}{h:>3}{r.mean:>10.4f}{r.t_naive:>9.2f}"
                f"{r.t_nw:>8.2f}{r.boot.block_size:>6.0f}{r.t_boot:>9.2f}"
                f"{r.boot_se:>9.4f}{ci:>22}{bca:>22}"
            )

    lines.append("")
    lines.append("  读法:")
    lines.append("  - t_naive 把重叠样本当独立样本，期限越长越不可信。")
    lines.append("  - t_NW 和 t_boot 都修了时间自相关；两者差距大时说明")
    lines.append("    lag/块长选得不合适，别只挑一个顺眼的看。")
    lines.append("  - L 是 bootstrap 实际块长；block_method=pw-sb/pw-cb 时")
    lines.append("    由 Politis-White 按序列自身自相关自动选择。")
    lines.append("  - 百分位区间只看重采样分布；BCa 另外修正了中位数偏差")
    lines.append("    （z0）和偏度（刀切加速度 a）。两者差异大时优先看 BCa。")
    lines.append("  - 两种区间都跨过 0，基本可以当作这条因子在这组数据上")
    lines.append("    没有可辨识的方向。")
    lines.append("=" * 100)
    return "\n".join(lines)
