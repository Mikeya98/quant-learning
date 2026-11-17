"""
因子归因测试。

    python tests/test_factor_model.py

检查三件事：
  1. 因子收益构造有没有偷看未来（分组用 t-1 信息、收益用 t 日）
  2. 回归能不能把已知的 beta 还原出来
  3. 收益分解的恒等式成不成立
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.analysis.factor_model import (
    build_factor_returns,
    estimate_exposures,
    factor_contribution,
    format_factor_report,
)

PASS = 0
FAIL = 0


def check(cond, name):
    global PASS, FAIL
    if cond:
        print(f"  [OK]   {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}")
        FAIL += 1


def close_from_returns(rets: pd.DataFrame, start=100.0) -> pd.DataFrame:
    """由日收益率反推价格路径（首行是起点，不产生收益）。"""
    px = (1.0 + rets).cumprod() * start
    first = pd.DataFrame([np.full(rets.shape[1], start)],
                         index=[rets.index[0] - pd.Timedelta(days=1)],
                         columns=rets.columns)
    return pd.concat([first, px])


def main():
    rng = np.random.default_rng(11)

    # =================================================================
    # 1. MKT 就是等权收益
    # =================================================================
    print("── MKT factor ──")
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    cols = [f"S{i}" for i in range(4)]
    rets = pd.DataFrame(rng.normal(0, 0.01, (40, 4)), index=idx, columns=cols)
    close = close_from_returns(rets)
    fr = build_factor_returns(close)

    # close_from_returns 在最前面补了一行起点，所以 pct_change 的结果
    # 正好对应 rets 的 40 行
    mkt_expected = rets.mean(axis=1)
    check(np.allclose(fr["MKT"].dropna(), mkt_expected, atol=1e-12),
          "MKT = equal-weight universe return")

    # =================================================================
    # 2. SMB: 小市值组涨得多 → 因子为正
    # =================================================================
    print("\n── SMB factor ──")
    # 4 只股票，市值恒定: 前两只小、后两只大
    cap = pd.DataFrame(1.0, index=idx, columns=cols)
    cap[cols[0]] = 1e9
    cap[cols[1]] = 2e9
    cap[cols[2]] = 1e11
    cap[cols[3]] = 2e11

    rets2 = pd.DataFrame(0.0, index=idx, columns=cols)
    rets2[cols[0]] = 0.02      # 小盘 +2%
    rets2[cols[1]] = 0.02
    rets2[cols[2]] = 0.00      # 大盘 0%
    rets2[cols[3]] = 0.00

    close2 = close_from_returns(rets2)
    fr2 = build_factor_returns(close2, mktcap=cap)

    check(abs(fr2["SMB"].iloc[-1] - 0.02) < 1e-12,
          "SMB = small(+2%) - big(0%) = +2%")

    # 分组的市值必须是 t-1 的：把最后一天的市值改成倒过来，
    # 当天因子不变
    cap_future = cap.copy()
    cap_future.loc[idx[-1], cols[0]] = 1e15   # 小盘变超大盘，但这是 t 日收盘后才知道的
    cap_future.loc[idx[-1], cols[2]] = 1e8
    fr3 = build_factor_returns(close2, mktcap=cap_future)
    check(abs(fr3["SMB"].iloc[-1] - fr2["SMB"].iloc[-1]) < 1e-12,
          "SMB uses t-1 market cap (no look-ahead)")

    # =================================================================
    # 3. MOM: 过去 5 日涨得多的组继续涨 → 因子为正
    # =================================================================
    print("\n── MOM factor ──")
    rets3 = pd.DataFrame(0.0, index=idx, columns=cols)
    # 前 6 天: A/B 涨, C/D 跌; 之后所有股票都按之前的强弱继续
    rets3.iloc[:6, 0] = 0.02
    rets3.iloc[:6, 1] = 0.02
    rets3.iloc[:6, 2] = -0.02
    rets3.iloc[:6, 3] = -0.02
    rets3.iloc[6:, 0] = 0.01
    rets3.iloc[6:, 1] = 0.01
    rets3.iloc[6:, 2] = -0.01
    rets3.iloc[6:, 3] = -0.01

    close3 = close_from_returns(rets3)
    fr4 = build_factor_returns(close3, mom_window=5)
    mom_valid = fr4["MOM"].dropna()
    check(len(mom_valid) > 0 and abs(mom_valid.iloc[-1] - 0.02) < 1e-12,
          "MOM = past winners(+1%) - losers(-1%) = +2%")

    # 翻转未来收益不影响过去
    check(fr4["MOM"].dropna().iloc[0] > 0,
          "early MOM is positive when winners keep winning")

    # =================================================================
    # 4. 回归能还原已知 beta
    # =================================================================
    print("\n── regression ──")
    rng = np.random.default_rng(2024)   # 独立随机源，避免受前面用例影响
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    f = pd.DataFrame({
        "MKT": rng.normal(0.0003, 0.01, n),
        "SMB": rng.normal(0.0000, 0.005, n),
        "MOM": rng.normal(0.0001, 0.004, n),
    }, index=idx)

    true_alpha = 0.0002
    true_beta = {"MKT": 1.2, "SMB": -0.4, "MOM": 0.7}
    y = (true_alpha
         + true_beta["MKT"] * f["MKT"]
         + true_beta["SMB"] * f["SMB"]
         + true_beta["MOM"] * f["MOM"])

    model = estimate_exposures(y, f)
    check(np.allclose(model.betas.values, [1.2, -0.4, 0.7], atol=1e-8),
          "recovers beta = 1.2 / -0.4 / 0.7")
    check(abs(model.alpha_daily - true_alpha) < 1e-10, "recovers alpha")
    check(abs(model.r2 - 1.0) < 1e-10, "R2 = 1 for exact fit")
    check(model.n_obs == n, "uses all observations")

    # 加噪声之后 beta 仍然接近
    noisy = y + pd.Series(rng.normal(0, 0.002, n), index=idx)
    model2 = estimate_exposures(noisy, f)
    check(np.allclose(model2.betas.values, [1.2, -0.4, 0.7], atol=0.05),
          "betas stay close with noise")
    check(model2.r2 < 0.999, "R2 drops when noise added")

    # =================================================================
    # 5. 收益分解恒等式
    # =================================================================
    print("\n── contribution identity ──")
    contrib = factor_contribution(model2)
    total = contrib.attrs["total_return"]
    check(abs(contrib["contribution"].sum() - total) < 1e-12,
          "alpha + factors + residual = total return")
    check(abs(contrib.attrs["identity_error"]) < 1e-12,
          "identity error < 1e-12")

    mkt_row = contrib[contrib["item"] == "MKT"].iloc[0]
    expected = float((model2.betas["MKT"] * f["MKT"]).sum())
    check(abs(mkt_row["contribution"] - expected) < 1e-12,
          "MKT contribution = beta × sum(factor return)")

    # =================================================================
    # 6. SMB 不可用时自动降级
    # =================================================================
    print("\n── missing size data ──")
    fr_no_size = pd.DataFrame({
        "MKT": f["MKT"],
        "SMB": np.nan,
        "MOM": f["MOM"],
    }, index=idx)
    model3 = estimate_exposures(y, fr_no_size)
    check(list(model3.betas.index) == ["MKT", "MOM"],
          "all-NaN SMB is dropped automatically")

    # =================================================================
    # 7. 样本外分解：残差可以不为 0
    # =================================================================
    print("\n── out-of-sample attribution ──")
    from src.analysis.factor_model import out_of_sample_attribution

    # 后半段加入因子解释不了的特异收益
    extra = pd.Series(0.0, index=idx)
    extra.iloc[n // 2:] = 0.001          # 后半段每天多赚 10bp
    y2 = y + extra
    model_oos, contrib_oos = out_of_sample_attribution(y2, f, split=0.5)

    check(contrib_oos.attrs["mode"] == "oos", "marked as out-of-sample")
    resid_row = contrib_oos[contrib_oos["item"] == "特异收益"].iloc[0]
    check(resid_row["contribution"] > 0.1,
          "OOS residual captures unexplained +10bp/day")
    check(abs(contrib_oos.attrs["identity_error"]) < 1e-12,
          "OOS decomposition also balances")

    # 样本内的残差和必然为 0
    in_sample = factor_contribution(model2)
    in_resid = in_sample[in_sample["item"] == "特异收益"].iloc[0]["contribution"]
    check(abs(in_resid) < 1e-12, "in-sample residual sums to zero")

    # =================================================================
    # 8. 报告能正常生成
    # =================================================================
    print("\n── report ──")
    text = format_factor_report(model2, contrib)
    check("因子归因" in text and "MKT" in text and "恒等式误差" in text,
          "report contains exposures and identity check")

    print(f"\n  Result: {PASS} passed, {FAIL} failed")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
