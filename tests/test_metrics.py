"""
基准相对指标测试。

不依赖 pytest，直接:

    python tests/test_metrics.py

用合成收益序列检查 beta / alpha / 跟踪误差 / 信息比率的定义，
重点是别把"基准涨了"当成"策略有 alpha"。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.analysis.metrics import (
    annualized_alpha,
    beta,
    information_ratio,
    tracking_error,
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


def series(values, start="20200101"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def main():
    rng = np.random.default_rng(7)
    n = 250
    bench = series(rng.normal(0.0004, 0.01, n))

    print("-- beta --")
    check(abs(beta(bench, bench) - 1.0) < 1e-9, "identical series: beta = 1")
    check(abs(beta(bench * 2, bench) - 2.0) < 1e-9, "2x series: beta = 2")
    check(abs(beta(bench * -0.5, bench) + 0.5) < 1e-9,
          "inverse half: beta = -0.5")
    flat = series(np.zeros(n))
    check(beta(bench, flat) == 0.0, "flat benchmark: beta = 0")

    print("-- alpha --")
    check(abs(annualized_alpha(bench, bench, risk_free_rate=0.0)) < 1e-9,
          "identical series: alpha = 0")
    check(abs(annualized_alpha(bench * 2, bench, risk_free_rate=0.0)) < 1e-9,
          "2x series: beta-adjusted alpha = 0")
    check(abs(annualized_alpha(bench + 0.001, bench, risk_free_rate=0.0)
              - 0.001 * 252 * 100) < 1e-9,
          "constant 0.1%/day excess: alpha = 25.2%/year")
    check(abs(annualized_alpha(bench, bench, risk_free_rate=0.03)) < 1e-9,
          "beta=1: alpha unchanged by risk-free rate")

    print("-- tracking error --")
    check(tracking_error(bench, bench) == 0.0, "identical series: TE = 0")
    check(tracking_error(bench + 0.001, bench) < 1e-9,
          "constant offset: TE = 0")

    print("-- information ratio --")
    check(information_ratio(bench, bench) == 0.0, "identical series: IR = 0")
    excess = series([0.01, -0.01] * 20)
    expected_ir = excess.mean() / excess.std() * np.sqrt(252)
    check(abs(information_ratio(bench[:40] + excess.values, bench[:40])
              - expected_ir) < 1e-9,
          "IR = mean(excess) / std(excess) * sqrt(252)")

    print("-- alignment --")
    a = series([0.01, 0.02, 0.03], start="20200101")
    b = series([0.01, 0.02, 0.03], start="20200102")
    # 只有 2020-01-02、01-03 两天重叠，收益率完全一样
    check(abs(beta(a, b) - 1.0) < 1e-9, "different dates are aligned first")
    c = series([0.01, 0.02, 0.03], start="20210101")
    check(beta(a, c) == 0.0, "no overlapping dates: beta = 0")

    print()
    print("=" * 60)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
