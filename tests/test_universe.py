"""
股票池模块的离线测试。

只测解析和过滤逻辑 —— 网络部分（requests）不在这里跑，
用样例 JSON 代替接口返回：

    python tests/test_universe.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.data.universe import parse_snapshot, filter_universe, parse_industry

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


SAMPLE = {
    "rc": 0,
    "data": {
        "total": 5,
        "diff": [
            {"f12": "600519", "f14": "贵州茅台", "f20": 2.1e12, "f21": 2.1e12},
            {"f12": "1", "f14": "平安银行", "f20": 2.5e11, "f21": 2.4e11},
            {"f12": "830799", "f14": "北交所某股", "f20": 1e9, "f21": 1e9},
            {"f12": "000002", "f14": "ST某某", "f20": 5e9, "f21": 5e9},
            {"f12": "300750", "f14": "宁德时代", "f20": 1.2e12, "f21": 1.0e12},
        ],
    },
}


def main():
    print("=" * 62)
    print("  股票池解析测试")
    print("=" * 62)

    print("\n── parse_snapshot ──")
    df = parse_snapshot(SAMPLE)
    check(len(df) == 5, "解析出 5 行")
    check(list(df.columns) == ["symbol", "name", "total_mktcap", "float_mktcap"],
          "列名正确")
    check(df.loc[df["name"] == "平安银行", "symbol"].iloc[0] == "000001",
          "代码补零到 6 位")
    check(abs(df.loc[df["name"] == "贵州茅台", "float_mktcap"].iloc[0] - 2.1e12) < 1,
          "市值字段正确")

    print("\n── parse_snapshot 空数据 ──")
    empty = parse_snapshot({"rc": 0, "data": None})
    check(empty.empty, "data=None 时返回空表")
    empty2 = parse_snapshot({})
    check(empty2.empty, "缺字段时返回空表")

    print("\n── filter_universe ──")
    pool = filter_universe(df)
    names = set(pool["name"])
    check("ST某某" not in names, "过滤 ST")
    check("北交所某股" not in names, "过滤北交所 (8 开头)")
    check(len(pool) == 3, "剩下 3 只")
    check(pool["float_mktcap"].is_monotonic_decreasing, "按流通市值降序")
    check(pool["symbol"].str.len().eq(6).all(), "代码都是 6 位")

    print("\n── filter_universe 保留 ST ──")
    kept = filter_universe(df, exclude_st=False)
    check("ST某某" in set(kept["name"]), "exclude_st=False 时保留 ST")

    print("\n── parse_industry ──")
    check(parse_industry({"data": {"f127": "酿酒行业"}}) == "酿酒行业", "取 f127")
    check(parse_industry({"data": {"f127": "-"}}) == "-", "占位符原样返回")
    check(parse_industry({"data": None}) is None, "无 data 返回 None")
    check(parse_industry({}) is None, "空 payload 返回 None")

    print("\n" + "=" * 62)
    print(f"  结果: {PASS} passed, {FAIL} failed")
    print("=" * 62)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
