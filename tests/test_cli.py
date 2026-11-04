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


def main():
    test_portfolio_cli_wiring()
    print("\n" + "=" * 40)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 40)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
