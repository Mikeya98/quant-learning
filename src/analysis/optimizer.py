"""
参数优化 —— 寻找策略的最佳参数组合。

方法：
1. Grid Search: 穷举参数空间，找样本内最优
2. Walk-Forward: 滚动训练/测试，评估样本外稳定性

⚠️ 核心警告（这个阶段最重要的认知）：
网格搜索找到的最优参数 ≈ 对历史数据的过拟合。
样本内夏普 0.8 → 样本外可能是 -0.2。
这就是为什么参数优化在量化里是个危险的工具而非圣杯。
"""

import itertools
import pandas as pd
import numpy as np

from ..engine.backtest import BacktestEngine


def grid_search(
    data: pd.DataFrame,
    strategy_cls,
    param_grid: dict,
    initial_capital: float = 100000.0,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """
    网格搜索 —— 穷举参数组合，按指定指标排序。

    Args:
        data:             OHLCV DataFrame
        strategy_cls:     策略类（未实例化）
        param_grid:       参数空间，如 {"fast": [5, 10, 15], "slow": [20, 30, 60]}
        initial_capital:  初始资金
        metric:           排序指标: "sharpe", "total_return", "max_dd", "win_rate"

    Returns:
        DataFrame，每行一组参数 + 绩效指标，按 metric 降序排列
    """
    from .metrics import compute_all

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    results = []

    total = 1
    for v in values:
        total *= len(v)

    n = 0
    data_len = len(data)
    for combo in itertools.product(*values):
        n += 1
        params = dict(zip(keys, combo))

        # 跳过窗口参数超过数据长度的组合
        max_window = max((v for v in params.values() if isinstance(v, int)), default=0)
        if max_window >= data_len:
            continue

        try:
            strategy = strategy_cls(**params)
            signals = strategy.run(data)
        except (ValueError, Exception):
            continue

        engine = BacktestEngine(initial_capital=initial_capital)
        result = engine.run(data, signals)

        if len(result.trades_df) == 0:
            continue

        m = compute_all(result)
        results.append({
            **params,
            "total_return_pct": m["total_return_pct"],
            "annual_return_pct": m["annual_return_pct"],
            "sharpe": m["sharpe_ratio"],
            "max_dd_pct": m["max_drawdown_pct"],
            "win_rate_pct": m["win_rate_pct"],
            "profit_factor": m["profit_factor"],
            "n_trades": m["n_trades"],
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # 按指定指标排序（夏普、收益等越高越好，回撤越低越好）
    ascending = metric in ("max_dd",)
    sort_key = "max_dd_pct" if metric == "max_dd" else metric
    df = df.sort_values(sort_key, ascending=ascending).reset_index(drop=True)

    return df


def walk_forward(
    data: pd.DataFrame,
    strategy_cls,
    param_grid: dict,
    train_window: int = 504,     # 2 年训练
    test_window: int = 126,      # 半年测试
    initial_capital: float = 100000.0,
    metric: str = "sharpe",
) -> pd.DataFrame:
    """
    滚动窗口优化 —— 样本内训练 → 样本外验证。

    每一步：
    1. 用 train_window 天数据做网格搜索，找最优参数
    2. 用最优参数在接下来的 test_window 天跑回测
    3. 窗口向前滚动 test_window 天

    Args:
        data:          OHLCV DataFrame
        strategy_cls:  策略类
        param_grid:    参数空间
        train_window:  训练窗口（交易日数）
        test_window:   测试窗口（交易日数）
        metric:        选择最优参数的指标

    Returns:
        DataFrame，每行一个窗口的训练/测试结果
    """
    from .metrics import compute_all

    n = len(data)
    results = []
    start = 0

    # 过滤掉在测试窗口装不下的参数
    safe_grid = {}
    for k, vals in param_grid.items():
        safe_grid[k] = [v for v in vals if v < test_window]
    if not any(safe_grid.values()):
        print("参数值全部超过测试窗口长度，无法执行")
        return pd.DataFrame()

    fold = 1
    while start + train_window + test_window <= n:
        train_data = data.iloc[start:start + train_window]
        test_data = data.iloc[start + train_window:start + train_window + test_window]

        # 样本内网格搜索
        gs = grid_search(train_data, strategy_cls, safe_grid,
                         initial_capital=initial_capital, metric=metric)
        if len(gs) == 0:
            start += test_window
            continue

        best = gs.iloc[0]
        best_params = {k: best[k] for k in param_grid.keys()}

        # 样本外测试
        try:
            strategy = strategy_cls(**best_params)
            test_signals = strategy.run(test_data)
            engine = BacktestEngine(initial_capital=initial_capital)
            test_result = engine.run(test_data, test_signals)
            test_metrics = compute_all(test_result)
        except (ValueError, Exception):
            # 最优参数在测试窗口不适用（如窗口参数过大），跳过
            start += test_window
            continue

        train_start = train_data["date"].iloc[0].date()
        train_end = train_data["date"].iloc[-1].date()
        test_start = test_data["date"].iloc[0].date()
        test_end = test_data["date"].iloc[-1].date()

        results.append({
            "fold": fold,
            "train_period": f"{train_start} ~ {train_end}",
            "test_period": f"{test_start} ~ {test_end}",
            **{f"best_{k}": v for k, v in best_params.items()},
            "train_sharpe": best["sharpe"],
            "train_return": best["total_return_pct"],
            "test_sharpe": test_metrics["sharpe_ratio"],
            "test_return": test_metrics["total_return_pct"],
            "test_max_dd": test_metrics["max_drawdown_pct"],
            "test_trades": test_metrics["n_trades"],
        })

        fold += 1
        start += test_window

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # 汇总统计
    avg_train_sharpe = df["train_sharpe"].mean()
    avg_test_sharpe = df["test_sharpe"].mean()
    print(f"\n  样本内平均夏普: {avg_train_sharpe:.2f}")
    print(f"  样本外平均夏普: {avg_test_sharpe:.2f}")
    if avg_train_sharpe > 0:
        decay = (avg_train_sharpe - avg_test_sharpe) / avg_train_sharpe * 100
        print(f"  衰减比例:       {decay:.0f}%")

    return df
