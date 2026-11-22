"""
命令行参数到回测器的传递测试。

重点不是回测算法，而是 `--exec` / `--st-file` 这两个参数有没有真的
传进 MultiAssetBacktester。之前它们只加进了 argparse 的参数表，
构造回测器时漏了，命令行看起来支持次日开盘，实际一直按收盘成交跑。

python tests/test_cli.py
"""

import contextlib
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import main as cli

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


def fake_data():
    """震荡行情，保证 MA2/MA3 会反复交叉。"""
    n = 120
    dates = pd.bdate_range("2020-01-01", periods=n)
    t = np.arange(n)
    close = 10.0 + 0.8 * np.sin(t / 4.0) + 0.002 * t
    open_ = close * (1.0 + 0.001 * np.cos(t / 3.0))
    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": np.maximum(open_, close) * 1.01,
        "low": np.minimum(open_, close) * 0.99,
        "close": close,
        "volume": 1e7,
        "amount": 1e9,
    })


def run_cli(argv):
    buf = io.StringIO()
    old_load = cli.load_csv
    cli.load_csv = lambda symbol, processed=True: fake_data()
    try:
        parser = cli.build_parser()
        args = parser.parse_args(argv)
        with contextlib.redirect_stdout(buf):
            result = cli.cmd_portfolio(args)
    finally:
        cli.load_csv = old_load
    return result, buf.getvalue()


def test_portfolio_cli_wiring():
    print("\n== portfolio CLI 参数传递 ==")

    fd, st_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(fd, "w") as f:
            f.write("600000\n")

        result, out = run_cli([
            "portfolio",
            "--symbols", "600000,600519",
            "--strategy", "ma_cross",
            "-p", "fast=2",
            "-p", "slow=3",
            "--rebalance", "1",
            "--max-participation", "0",
            "--exec", "next_open",
            "--st-file", st_path,
        ])

        check(result is not None, "cmd_portfolio returns a result")
        if result is None:
            return

        check(result.config.get("exec_mode") == "next_open",
              "--exec next_open reaches the backtester")
        check(result.config.get("st_symbols") == 1,
              "--st-file reaches the backtester")
        check("次日开盘" in out, "report header shows next-open mode")

        buys = result.trades_df[result.trades_df["side"] == "BUY"]
        check(len(buys) > 0, "synthetic data produces buys")
        if len(buys):
            row = buys.iloc[0]
            # 成交价 = 当日开盘 × (1 + 滑点)，不再等于收盘价
            expected = row["price"] / 1.001
            day = fake_data()
            open_of_day = day.loc[day["date"] == row["date"], "open"].iloc[0]
            check(abs(expected - open_of_day) < 0.02,
                  "next_open buy price uses the open, not the close")
    finally:
        try:
            os.unlink(st_path)
        except OSError:
            pass


def test_factor_portfolio_flags():
    print("\n== factor-portfolio 预处理参数 ==")

    parser = cli.build_parser()
    args = parser.parse_args([
        "factor-portfolio",
        "--symbols", "000001,600036",
        "--winsorize",
        "--neutralize", "industry,size",
        "--orthogonalize",
        "--industry-file", "data/meta/industry.csv",
    ])

    check(args.winsorize is True, "--winsorize 传进参数表")
    check(args.neutralize == "industry,size", "--neutralize 原样保留")
    check(args.orthogonalize is True, "--orthogonalize 传进参数表")
    check(args.industry_file == "data/meta/industry.csv",
          "--industry-file 传进参数表")

    # 默认值：不预处理
    args2 = parser.parse_args([
        "factor-portfolio", "--symbols", "000001,600036",
    ])
    check(args2.winsorize is False and args2.neutralize == ""
          and args2.orthogonalize is False,
          "默认关闭所有预处理")


def test_factor_tune_flags():
    print("\n== factor-tune 参数 ==")
    parser = cli.build_parser()

    args = parser.parse_args([
        "factor-tune",
        "--symbols", "000001,000002,600036,600519,601318",
        "--grid-periods", "5,20",
        "--grid-windows", "40,120",
        "--folds", "4",
        "--mode", "rolling",
        "--criterion", "icir",
        "--initial-train", "0.6",
    ])
    check(args.command == "factor-tune", "命令识别")
    check(args.grid_periods == "5,20" and args.grid_windows == "40,120",
          "网格参数原样保留")
    check(args.folds == 4 and args.mode == "rolling", "折数/模式传进参数表")
    check(args.criterion == "icir", "挑选标准传进参数表")
    check(abs(args.initial_train - 0.6) < 1e-12, "训练段占比传进参数表")

    args2 = parser.parse_args([
        "factor-tune", "--symbols", "000001,000002,600036,600519,601318",
    ])
    check(args2.folds == 3 and args2.mode == "expanding"
          and args2.criterion == "ic",
          "默认 3 折 / expanding / IC")
    check(args2.grid_periods == "5,10,20,40", "默认周期网格")
    check(args2.period == 5, "默认评估期 5 日")


def test_factor_sensitivity_flags():
    print("\n== factor-sensitivity 参数 ==")
    parser = cli.build_parser()

    args = parser.parse_args([
        "factor-sensitivity",
        "--symbols", "000001,000002,600036,600519,601318",
        "--horizons", "1,5,20",
        "--lookbacks", "5,20",
        "--kind", "reversal",
        "--segments", "3",
        "--stability-horizon", "10",
    ])
    check(args.command == "factor-sensitivity", "命令识别")
    check(args.horizons == "1,5,20" and args.lookbacks == "5,20",
          "horizon/lookback 原样保留")
    check(args.kind == "reversal", "因子方向传进参数表")
    check(args.segments == 3 and args.stability_horizon == 10,
          "分段/稳定性期限传进参数表")

    args2 = parser.parse_args([
        "factor-sensitivity", "--symbols", "000001,000002,600036,600519,601318",
    ])
    check(args2.horizons == "1,3,5,10,20", "默认评估期限")
    check(args2.lookbacks == "5,10,20,40,60", "默认信号期限")
    check(args2.kind == "momentum" and args2.segments == 4,
          "默认动量方向 / 4 段")

    # 端到端跑一遍（数据用假行情，只验证接线和输出）
    buf = io.StringIO()
    old_load = cli.load_csv
    cli.load_csv = lambda symbol, processed=True: fake_data()
    try:
        args3 = parser.parse_args([
            "factor-sensitivity",
            "--symbols", "600000,600519,000001,000002,601318",
            "--horizons", "1,5", "--lookbacks", "5,20",
        ])
        with contextlib.redirect_stdout(buf):
            cli.cmd_factor_sensitivity(args3)
    finally:
        cli.load_csv = old_load
    out = buf.getvalue()
    check("IC 期限敏感性" in out and "Newey-West" in out,
          "cmd_factor_sensitivity 输出完整报告")


def main():
    test_portfolio_cli_wiring()
    test_factor_portfolio_flags()
    test_factor_tune_flags()
    test_factor_sensitivity_flags()
    print("\n" + "=" * 40)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 40)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
