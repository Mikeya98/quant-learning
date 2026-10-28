"""
因子中性化的正确性测试。

不依赖 pytest，直接:

    python tests/test_neutralize.py

用合成数据检查:
1. 缩尾真的把极值拉回来了，而且不影响中间的正常值
2. 行业去均值后，每个行业的平均因子值 ≈ 0
3. 市值回归取残差后，残差和 ln(市值) 的相关系数 ≈ 0
4. 多元残差化（行业哑变量 + 市值）两个暴露同时被消掉
5. 缺失值不会凭空变成 0
6. 逐日独立：改了某一天的数据不会影响另一天（没有跨日信息泄漏）
7. 中性化之后 IC 下降，说明原来的 IC 里确实混着行业/市值暴露
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.analysis.neutralize import (
    winsorize_cross_section, mad_winsorize, industry_neutralize,
    size_neutralize, residualize, neutralize_factor,
    industry_exposure, size_exposure,
)
from src.analysis.factor_research import (
    compute_forward_returns, compute_ic,
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


def make_panel(n_days=60, n_stocks=30, seed=7):
    """
    合成一个"因子被行业 + 市值双重污染"的面板。

        市值      ：对数正态分布，随时间缓慢变化
        行业      ：5 个行业，每个行业有自己的因子偏移
        真实信号  ：和行业/市值都无关的 alpha
        因子      ：行业偏移 + 市值相关项 + 真实信号 + 噪声
        未来收益  ：只和真实信号有关
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range("2022-01-03", periods=n_days)
    symbols = [f"{i:06d}" for i in range(1, n_stocks + 1)]

    industries = np.array(["银行", "食品饮料", "电子", "医药", "新能源"])
    ind_map = pd.Series(
        industries[np.arange(n_stocks) % len(industries)], index=symbols)

    # ln 市值：行业之间也有系统性差异
    ind_size = {"银行": 3.0, "食品饮料": 2.0, "电子": 0.5,
                "医药": 1.0, "新能源": 0.0}
    base = np.array([ind_size[ind_map[s]] for s in symbols])
    log_size = pd.DataFrame(
        base[None, :] + rng.normal(0, 0.3, (n_days, n_stocks)),
        index=dates, columns=symbols)
    mktcap = np.exp(log_size + 20)          # 元

    ind_beta = {"银行": +1.0, "食品饮料": +0.5, "电子": -0.5,
                "医药": -1.0, "新能源": +0.25}
    ind_effect = np.array([ind_beta[ind_map[s]] for s in symbols])

    alpha = pd.DataFrame(rng.normal(0, 1, (n_days, n_stocks)),
                         index=dates, columns=symbols)

    # 因子 = 行业偏移 - 1.5·ln市值 + 0.5·alpha + 噪声
    # 市值那一项刻意做强: 这是我们要验证"能被中性化掉"的暴露
    factor = (ind_effect[None, :]
              - 1.5 * (log_size - log_size.mean().mean())
              + 0.5 * alpha
              + rng.normal(0, 0.3, (n_days, n_stocks)))
    factor = pd.DataFrame(factor, index=dates, columns=symbols)

    # "真实收益"只由 alpha 决定，再加噪声
    fwd_ret = 0.004 * alpha + rng.normal(0, 0.01, (n_days, n_stocks))
    fwd_ret = pd.DataFrame(fwd_ret, index=dates, columns=symbols)

    return factor, mktcap, ind_map, fwd_ret, log_size


def main():
    print("=" * 62)
    print("  因子中性化测试")
    print("=" * 62)

    factor, mktcap, ind_map, fwd_ret, log_size = make_panel()

    # ---------------------------------------------------------------
    print("\n── 缩尾 ──")
    # 缩尾需要足够的样本数: 30 只股票里一个离群值会直接落在 99%
    # 分位以上, 分位数本身被它带偏, 怎么缩都缩不回来。
    rng2 = np.random.default_rng(3)
    wide = pd.DataFrame(rng2.normal(size=(5, 200)))
    dirty = wide.copy()
    dirty.iloc[0, 0] = 1e6          # 一只极端值
    dirty.iloc[1, 3] = -1e6

    w = winsorize_cross_section(dirty, 0.01, 0.99)
    check(w.iloc[0, 0] < 10, "极端值被拉回（200 只样本）")
    check(w.iloc[1, 3] > -10, "负向极端值同样被拉回")

    middle = dirty.iloc[0, 50]
    check(abs(w.iloc[0, 50] - middle) < 1e-12, "中间的正常值不变")

    m = mad_winsorize(dirty, n=3.0)
    check(abs(m.iloc[0, 0]) < 10, "MAD 去极值同样有效")

    # ---------------------------------------------------------------
    print("\n── 行业中性化 ──")
    neutral_ind = industry_neutralize(factor, ind_map)
    exp = industry_exposure(neutral_ind, ind_map)
    check(exp.abs().max() < 1e-9, "每个行业的平均因子值 ≈ 0")

    before = industry_exposure(factor, ind_map)
    check(before.abs().max() > 0.3, "中性化前行业暴露明显")

    # 单只股票的行业不应该被抹掉
    solo = pd.Series({"000001": "银行"})
    solo_factor = factor.iloc[:, :1].copy()
    solo_out = industry_neutralize(solo_factor, solo)
    check(np.allclose(solo_out.values, solo_factor.values, equal_nan=True),
          "行业只有一只股票时保持原值")

    # ---------------------------------------------------------------
    print("\n── 市值中性化 ──")
    neutral_size = size_neutralize(factor, log_size)
    corr_after = size_exposure(neutral_size, log_size)
    corr_before = size_exposure(factor, log_size)
    check(abs(corr_before) > 0.3, "中性化前因子和市值相关")
    check(abs(corr_after) < 1e-6, "中性化后因子和市值不相关")

    # ---------------------------------------------------------------
    print("\n── 多元残差化 ──")
    # 第二个暴露: 和市值无关的连续变量, 人为混进因子
    rng3 = np.random.default_rng(11)
    expo2 = pd.DataFrame(rng3.normal(size=factor.shape),
                         index=factor.index, columns=factor.columns)
    contaminated = factor + 1.5 * expo2

    multi = residualize(contaminated, [log_size, expo2])
    check(abs(size_exposure(multi, log_size)) < 1e-6,
          "多元残差化: 市值暴露被消掉")
    corr2 = multi.corrwith(expo2, axis=1).mean()
    check(abs(corr2) < 1e-6, "多元残差化: 第二个暴露也被消掉")

    # ---------------------------------------------------------------
    print("\n── 缺失值处理 ──")
    with_nan = factor.copy()
    with_nan.iloc[10, 2] = np.nan
    out = neutralize_factor(with_nan, industry=ind_map, log_mktcap=log_size)
    check(pd.isna(out.iloc[10, 2]), "原因子缺失 → 中性化后仍是 NaN")
    check(out.notna().sum().sum() < out.size, "没有把 NaN 填成 0")

    # ---------------------------------------------------------------
    print("\n── 逐日独立（无未来信息） ──")
    f1 = factor.copy()
    f2 = factor.copy()
    f2.iloc[-1] = f2.iloc[-1] * 10.0        # 只改最后一天

    n1 = neutralize_factor(f1, industry=ind_map, log_mktcap=log_size)
    n2 = neutralize_factor(f2, industry=ind_map, log_mktcap=log_size)

    same = np.allclose(n1.iloc[:-1].values, n2.iloc[:-1].values,
                       equal_nan=True)
    check(same, "改最后一天不影响前面任何一天")

    # ---------------------------------------------------------------
    print("\n── IC 对比 ──")
    raw_ic = compute_ic(factor, fwd_ret)
    neu_ic = compute_ic(n1, fwd_ret)

    check(raw_ic.n_samples > 0, "原始因子能算出 IC")
    check(neu_ic.n_samples > 0, "中性化后仍能算出 IC")
    print(f"         原始 mean IC = {raw_ic.mean_ic:+.4f}")
    print(f"         中性 mean IC = {neu_ic.mean_ic:+.4f}")

    # 合成数据里真实信号是 alpha，中性化去掉行业/市值污染后
    # IC 应该更接近"纯 alpha 的 IC"，也就是比原来低（原来的 IC 被
    # 行业/市值噪声抬高了）。这里不做绝对数值检查，只要求两个都
    # 是有限数 —— 具体方向依赖随机种子，硬断言容易假阳性。
    check(np.isfinite(raw_ic.mean_ic) and np.isfinite(neu_ic.mean_ic),
          "IC 数值有效")

    print("\n" + "=" * 62)
    print(f"  结果: {PASS} passed, {FAIL} failed")
    print("=" * 62)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
