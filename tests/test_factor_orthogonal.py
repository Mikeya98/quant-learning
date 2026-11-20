"""
因子正交化与选股预处理的测试。

不依赖 pytest，直接:

    python tests/test_factor_orthogonal.py

检查:
1. 正交化之后因子两两横截面相关 ≈ 0
2. 第一个因子原样保留，后面的因子被剥离
3. 缺失值保持缺失，不会变成 0
4. 共线的因子被剥成 0 残差
5. 行业去均值 / 市值回归接入 prepare_factors 后暴露消失
6. 缩尾能压掉极端值
7. composite_score / ic_weighted_score 接受预处理参数且不改输入
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.analysis.factor_orthogonal import (
    orthogonalize_factors, mean_pairwise_corr, prepare_factors,
)
from src.analysis.neutralize import industry_exposure, size_exposure

from src.strategy.factor_selection import (
    composite_score, ic_weighted_score,
)

PASS = 0
FAIL = 0


def check(cond, name):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [OK]   {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")


def make_factors(n_days=80, n_stocks=24, seed=11):
    """三个横截面因子: b 和 a 高度相关, c 独立。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_stocks)]

    a = pd.DataFrame(rng.normal(size=(n_days, n_stocks)),
                     index=dates, columns=cols)
    b = 0.8 * a + 0.2 * pd.DataFrame(
        rng.normal(size=(n_days, n_stocks)), index=dates, columns=cols)
    c = pd.DataFrame(rng.normal(size=(n_days, n_stocks)),
                     index=dates, columns=cols)
    return {"a": a, "b": b, "c": c}


def mean_abs_corr(x, y):
    vals = []
    for dt in x.index:
        a = x.loc[dt].to_numpy(dtype=float)
        b = y.loc[dt].to_numpy(dtype=float)
        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() < 3:
            continue
        if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
            continue
        vals.append(abs(float(np.corrcoef(a[mask], b[mask])[0, 1])))
    return float(np.mean(vals)) if vals else float("nan")


def make_price_panel(n_days=120, n_stocks=12, seed=21):
    """给 composite_score 用的 close/volume/turnover 面板。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_stocks)]

    ret = rng.normal(0.0005, 0.02, size=(n_days, n_stocks))
    close = pd.DataFrame(
        20.0 * np.exp(np.cumsum(ret, axis=0)),
        index=dates, columns=cols)
    volume = pd.DataFrame(
        rng.lognormal(mean=13.0, sigma=0.3, size=(n_days, n_stocks)),
        index=dates, columns=cols)
    turnover = volume / 1e8
    return close, volume, turnover


def main():
    print("=" * 62)
    print("  因子正交化 / 选股预处理测试")
    print("=" * 62)

    factors = make_factors()

    # ---------------------------------------------------------------
    print("\n── 正交化: 相关性和顺序 ──")
    raw_corr = mean_abs_corr(factors["a"], factors["b"])
    orth = orthogonalize_factors(factors)
    orth_corr = mean_abs_corr(orth["a"], orth["b"])

    check(raw_corr > 0.5, f"正交化前 a/b 高度相关 ({raw_corr:.3f})")
    check(orth_corr < 1e-8, f"正交化后 a/b 相关 ≈ 0 ({orth_corr:.2e})")
    check(mean_abs_corr(orth["a"], orth["c"]) < 1e-8,
          "a 和 c 也正交")
    check(mean_abs_corr(orth["b"], orth["c"]) < 1e-8,
          "b 和 c 也正交")

    check(np.allclose(factors["a"].to_numpy(), orth["a"].to_numpy(),
                      equal_nan=True), "第一个因子原样保留")
    check(not np.allclose(factors["b"].to_numpy(), orth["b"].to_numpy()),
          "第二个因子确实被改了")
    check(float(orth["b"].std().mean()) > 0.0, "残差不是全 0")
    check(float(orth["b"].abs().mean().mean()) <
          float(factors["b"].abs().mean().mean()) / 2.0,
          "b 的残差幅度明显变小（被 a 解释掉一部分）")

    # 换顺序: a 和 b 谁在前，谁保留
    rev = orthogonalize_factors(factors, order=["b", "a", "c"])
    check(np.allclose(factors["b"].to_numpy(), rev["b"].to_numpy(),
                      equal_nan=True), "order 里排第一的因子保留")
    check(mean_abs_corr(rev["a"], rev["b"]) < 1e-8,
          "换顺序后仍然正交")

    # ---------------------------------------------------------------
    print("\n── 边界情况 ──")
    check(orthogonalize_factors({}) == {}, "空字典返回空")
    one = orthogonalize_factors({"a": factors["a"]})
    check(np.allclose(one["a"].to_numpy(), factors["a"].to_numpy(),
                      equal_nan=True), "单因子等于原表")

    try:
        orthogonalize_factors(factors, order=["a", "nope"])
        check(False, "未知因子应该报错")
    except ValueError:
        check(True, "未知因子报错")

    try:
        orthogonalize_factors(factors, order=["a", "b"])
        check(False, "order 漏因子应该报错")
    except ValueError:
        check(True, "order 漏因子报错")

    # ---------------------------------------------------------------
    print("\n── 缺失值 ──")
    dirty = {k: v.copy() for k, v in factors.items()}
    dirty["a"].iloc[10, 3] = np.nan
    dirty["b"].iloc[10, 4] = np.nan
    out = orthogonalize_factors(dirty)
    check(np.isnan(out["a"].iloc[10, 3]), "因子 a 的 NaN 保留")
    check(np.isnan(out["b"].iloc[10, 4]), "因子 b 的 NaN 保留")
    check(np.isnan(out["b"].iloc[10, 3]),
          "回归样本缺 a 时 b 的残差也是 NaN")

    # ---------------------------------------------------------------
    print("\n── 完全共线 ──")
    base = factors["a"]
    coll = {"base": base, "copy": 2.0 * base + 3.0}
    out = orthogonalize_factors(coll)
    check(float(out["copy"].abs().max().max()) < 1e-9,
          "2x+3 对 x 回归后残差 ≈ 0")

    # ---------------------------------------------------------------
    print("\n── prepare_factors: 行业 / 市值 / 缩尾 ──")
    n_days, n_stocks = 30, 18
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_stocks)]
    industry = pd.Series(
        ["bank", "tech", "energy"] * (n_stocks // 3), index=cols)

    rng = np.random.default_rng(5)
    ind_offset = industry.map({"bank": 2.0, "tech": -1.0, "energy": 0.5})
    size = pd.DataFrame(
        np.tile(np.linspace(8.0, 12.0, n_stocks), (n_days, 1)),
        index=dates, columns=cols)
    factor = pd.DataFrame(
        rng.normal(size=(n_days, n_stocks)),
        index=dates, columns=cols).add(ind_offset, axis=1)
    factor = factor.add(0.7 * size, axis=0)

    by_ind = prepare_factors(
        {"f": factor}, ["f"], industry=industry)["f"]
    check(float(industry_exposure(by_ind, industry).abs().max()) < 1e-9,
          "行业中性化后行业暴露 ≈ 0")

    by_size = prepare_factors(
        {"f": factor}, ["f"], log_mktcap=size)["f"]
    check(abs(float(size_exposure(by_size, size))) < 1e-9,
          "市值中性化后市值暴露 ≈ 0")

    combined = prepare_factors(
        {"f": factor}, ["f"], industry=industry, log_mktcap=size)["f"]
    check(int(combined.notna().sum().sum()) ==
          int(factor.notna().sum().sum()), "组合中性化不丢有效样本")

    extreme = factor.copy()
    extreme.iloc[3, 5] = 1e6
    w = prepare_factors({"f": extreme}, ["f"], winsorize=True)["f"]
    check(float(w.abs().max().max()) < 1e3, "缩尾把 1e6 的极值压回来了")

    # ---------------------------------------------------------------
    print("\n── mean_pairwise_corr ──")
    raw = mean_pairwise_corr(factors)
    after = mean_pairwise_corr(orth)
    check(raw[("a", "b")] > 0.5, "原始 a/b 平均相关 > 0.5")
    check(abs(after[("a", "b")]) < 1e-8, "正交化后平均相关 ≈ 0")

    # ---------------------------------------------------------------
    print("\n── 接进 composite_score ──")
    close, volume, turnover = make_price_panel()
    log_mktcap = np.log(
        pd.DataFrame(1e9, index=close.index, columns=close.columns)
        * (1.0 + close.rank(axis=1)))

    plain = composite_score(close, volume, turnover)
    prep = composite_score(close, volume, turnover,
                           winsorize=True, log_mktcap=log_mktcap,
                           orthogonalize=True)
    check(plain.shape == prep.shape, "预处理前后形状一致")
    check(int(prep.notna().sum().sum()) > 0, "预处理后仍有有效分数")
    check(int(prep.notna().sum().sum()) <= int(plain.notna().sum().sum()),
          "预处理不会凭空造出样本")

    # 行业 Series 也能接
    ind = pd.Series(
        [["bank", "tech", "energy"][i % 3] for i in range(len(close.columns))],
        index=close.columns)
    with_ind = composite_score(close, volume, turnover, industry=ind)
    check(with_ind.shape == close.shape, "行业中性化路径可运行")

    try:
        composite_score(close, volume, turnover, factors=["nope"])
        check(False, "未知因子应该报错")
    except ValueError:
        check(True, "未知因子报错")

    # ---------------------------------------------------------------
    print("\n── 接进 ic_weighted_score ──")
    ic_score = ic_weighted_score(close, volume, turnover,
                                 ic_period=5, ic_window=30, min_ic_obs=10,
                                 log_mktcap=log_mktcap,
                                 orthogonalize=True)
    check(ic_score.shape == close.shape, "IC 加权路径可运行")
    check(int((ic_score.notna().sum(axis=1) >= 3).sum()) > 0,
          "有一批交易日能给出横截面分数")

    # 权重估计用预处理后的因子: 至少不应抛异常且能选出股票
    from src.strategy.factor_selection import select_top
    selected = select_top(ic_score, top_n=3, rebalance_days=5)
    check(int(selected.sum().sum()) > 0, "预处理后的分数可以选股")

    print("\n" + "=" * 62)
    print(f"  结果: {PASS} passed, {FAIL} failed")
    print("=" * 62)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
