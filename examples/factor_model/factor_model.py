"""
因子模型演示 —— Fama-French 三因子模型的完整 Python 实现。

把 CAPM 的单因子（市场）扩展到三因子：
    R_i - R_f = alpha + beta_m*MKT + beta_s*SMB + beta_h*HML + epsilon

MKT = 市场超额收益（市场组合 - 无风险利率）
SMB = 小市值组合 - 大市值组合（Small Minus Big）
HML = 高账面市值比 - 低账面市值比（High Minus Low）

输出：
    1. 三因子累计收益曲线图 → data/charts/factor_cumulative_returns.png
    2. 示例股票的因子暴露表格
    3. 横截面验证（因子暴露 vs 实际收益）

运行：
    python examples/factor_model/factor_model.py
"""

import sys
import io
# Windows 下强制 UTF-8 输出，避免 GBK 编码错误
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
N_STOCKS = 30                # 模拟股票数量
N_DAYS = 504                 # 大约 2 年交易日（252 × 2）
RISK_FREE_RATE = 0.025       # 年化无风险利率 2.5%
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


# ===================================================================
# 第一部分：模拟 A 股风格的股票数据
# ===================================================================
def simulate_stock_universe(n_stocks: int, n_days: int) -> pd.DataFrame:
    """
    生成模拟的 A 股股票日线数据。

    模拟逻辑：
    - 每只股票有一个"真实"的因子暴露（beta_smb, beta_hml），这决定了它对因子的敏感度
    - 股票收益率 = β_mkt×MKT + β_smb×SMB + β_hml×HML + ε（特质噪声）
    - 市值和 PB 也模拟出来，体现规模和价值特征

    返回 DataFrame 列：
        date, stock_code, close_prev, close, daily_ret, market_cap, pb_ratio
    """
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")

    # -- 先生成三因子日收益率（年化波动率 ~15%） --
    daily_vol = 0.15 / np.sqrt(252)
    MKT = np.random.normal(0.0003, daily_vol, n_days)       # 年化 ~5% 期望
    SMB = np.random.normal(0.0002, daily_vol * 0.6, n_days)  # 规模因子
    HML = np.random.normal(0.0001, daily_vol * 0.5, n_days)  # 价值因子

    rows = []
    for i in range(n_stocks):
        code = f"{600000 + i:06d}"  # 600000, 600001, ...

        # 每只股票的因子暴露
        beta_mkt = np.random.uniform(0.7, 1.3)
        beta_smb = np.random.normal(0.0, 0.8)  # 正 = 小盘股特征, 负 = 大盘股
        beta_hml = np.random.normal(0.0, 0.6)  # 正 = 价值股, 负 = 成长股

        # 特质噪声
        idio_vol = np.random.uniform(0.01, 0.03) / np.sqrt(252)
        epsilon = np.random.normal(0, idio_vol, n_days)

        # 股票日收益率
        daily_ret = beta_mkt * MKT + beta_smb * SMB + beta_hml * HML + epsilon

        # 模拟市值（对数正态分布）和 PB
        base_cap = 10 ** np.random.uniform(9, 11)  # 10亿 ~ 1000亿
        cap_noise = np.exp(np.cumsum(np.random.normal(0, 0.02, n_days)))
        market_cap = base_cap * cap_noise

        base_pb = np.random.uniform(0.8, 8.0)  # PB 范围
        pb_noise = np.exp(np.cumsum(np.random.normal(0, 0.01, n_days)))
        pb_ratio = base_pb * pb_noise

        # 价格（从 10 元出发）
        close = 10.0 * np.exp(np.cumsum(daily_ret))
        close_prev = np.concatenate([[10.0], close[:-1]])

        stock_df = pd.DataFrame({
            "date": dates,
            "stock_code": code,
            "close_prev": close_prev,
            "close": close,
            "daily_ret": daily_ret,
            "market_cap": market_cap,
            "pb_ratio": pb_ratio,
            "beta_smb_true": beta_smb,  # 真实暴露（用来验证回归是否准确）
            "beta_hml_true": beta_hml,
        })
        rows.append(stock_df)

    return pd.concat(rows, ignore_index=True)


# ===================================================================
# 第二部分：构建因子
# ===================================================================
def build_factors(data: pd.DataFrame) -> pd.DataFrame:
    """
    从股票池构建 Fama-French 三因子。

    分组方式：
    - SMB：每天按市值分 3 组（S=小, M=中, B=大），S 组等权收益 - B 组等权收益
    - HML：每天按 B/P（= 1/PB）分 3 组（H=高BP=价值, L=低BP=成长），
            H 组等权收益 - L 组等权收益

    注意：这里用 B/P 而不是 PB，因为 PB 越低 = 估值越低 = 价值股。
    如果用 PB 直接分组，方向会反。
    """
    dates = sorted(data["date"].unique())

    daily_mkt_ret = data.groupby("date")["daily_ret"].mean()
    factors = pd.DataFrame({
        "MKT": daily_mkt_ret - RISK_FREE_RATE / 252,
    }, index=dates)

    # 计算 B/P = 1 / PB
    data = data.copy()
    data["bp"] = 1.0 / data["pb_ratio"]

    # SMB 和 HML 按日期循环计算
    smb_list = []
    hml_list = []
    for d in dates:
        day_data = data[data["date"] == d]

        # 市值分组
        day_data["size_group"] = pd.qcut(
            day_data["market_cap"], 3, labels=["S", "M", "B"]
        )
        s_ret = day_data[day_data["size_group"] == "S"]["daily_ret"].mean()
        b_ret = day_data[day_data["size_group"] == "B"]["daily_ret"].mean()
        smb_list.append(s_ret - b_ret)

        # 价值分组
        day_data["value_group"] = pd.qcut(
            day_data["bp"], 3, labels=["L", "M", "H"]
        )
        h_ret = day_data[day_data["value_group"] == "H"]["daily_ret"].mean()
        l_ret = day_data[day_data["value_group"] == "L"]["daily_ret"].mean()
        hml_list.append(h_ret - l_ret)

    factors["SMB"] = smb_list
    factors["HML"] = hml_list

    return factors.dropna()


# ===================================================================
# 第三部分：因子暴露 —— 回归
# ===================================================================
def calc_exposure(returns: pd.Series, factors: pd.DataFrame) -> dict:
    """对单只股票做 OLS 回归，返回因子暴露。"""
    common_idx = returns.index.intersection(factors.index)
    y = returns.loc[common_idx].values
    X = sm.add_constant(factors.loc[common_idx, ["MKT", "SMB", "HML"]].values)

    model = sm.OLS(y, X).fit()
    return {
        "alpha_daily": model.params[0],
        "beta_mkt": model.params[1],
        "beta_smb": model.params[2],
        "beta_hml": model.params[3],
        "t_alpha": model.tvalues[0],
        "t_smb": model.tvalues[2],
        "t_hml": model.tvalues[3],
        "r2": model.rsquared,
    }


# ===================================================================
# 第四部分：可视化
# ===================================================================
def plot_factors(factors: pd.DataFrame, output_path: Path):
    """绘制三因子累计收益曲线"""
    cumulative = (1 + factors[["MKT", "SMB", "HML"]]).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    for col in ["MKT", "SMB", "HML"]:
        ax.plot(cumulative.index, cumulative[col], label=col, linewidth=1.2)

    ax.set_title("Fama-French 三因子累计收益曲线（模拟数据）", fontsize=13, fontweight="bold")
    ax.set_ylabel("累计净值 (1.0 = 起点)")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] 因子曲线图已保存 → {output_path}")


# ===================================================================
# 第五部分：横截面验证
# ===================================================================
def cross_section_check(data: pd.DataFrame, factors: pd.DataFrame):
    """
    验证：某一天，因子暴露高的股票，收益是否真的更高？

    思路：取最后一天，对每只股票算 beta_smb 和 beta_hml（用全期数据回归），
    然后比较：当天 SMB 为正时，高 beta_smb 的股票当天收益是否更高。
    """
    last_date = data["date"].max()
    last_day_ret = data[data["date"] == last_date][["stock_code", "daily_ret"]].set_index("stock_code")

    # 对每只股票做全期回归
    exposures = {}
    for code, group in data.groupby("stock_code"):
        ret_series = group.set_index("date")["daily_ret"]
        try:
            exposures[code] = calc_exposure(ret_series, factors)
        except Exception:
            continue

    exp_df = pd.DataFrame(exposures).T
    merged = exp_df.join(last_day_ret, how="inner")

    print(f"\n{'='*60}")
    print(f"横截面验证（{last_date.date()}）")
    print(f"当天因子收益: MKT={factors.loc[last_date, 'MKT']:.4%},  "
          f"SMB={factors.loc[last_date, 'SMB']:.4%},  "
          f"HML={factors.loc[last_date, 'HML']:.4%}")
    print(f"{'='*60}")

    # 按 beta_smb 分成高/低两组，看当天收益差
    merged["smb_group"] = pd.qcut(merged["beta_smb"], 2, labels=["低SMB暴露", "高SMB暴露"])
    smb_hi = merged[merged["smb_group"] == "高SMB暴露"]["daily_ret"].mean()
    smb_lo = merged[merged["smb_group"] == "低SMB暴露"]["daily_ret"].mean()

    print(f"\n按 SMB 暴露分组当天平均收益：")
    print(f"  高 SMB 暴露组: {smb_hi:.4%}")
    print(f"  低 SMB 暴露组: {smb_lo:.4%}")
    print(f"  差值          : {smb_hi - smb_lo:.4%}")

    # 同理 HML
    merged["hml_group"] = pd.qcut(merged["beta_hml"], 2, labels=["低HML暴露", "高HML暴露"])
    hml_hi = merged[merged["hml_group"] == "高HML暴露"]["daily_ret"].mean()
    hml_lo = merged[merged["hml_group"] == "低HML暴露"]["daily_ret"].mean()

    print(f"\n按 HML 暴露分组当天平均收益：")
    print(f"  高 HML 暴露组: {hml_hi:.4%}")
    print(f"  低 HML 暴露组: {hml_lo:.4%}")
    print(f"  差值          : {hml_hi - hml_lo:.4%}")

    print(f"\n解读：当天 SMB={factors.loc[last_date, 'SMB']:.4%}，"
          f"HML={factors.loc[last_date, 'HML']:.4%}。")
    print("如果 SMB>0 且高SMB暴露组收益更高 → 因子暴露确实在解释收益差异。")

    return exp_df


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print("Fama-French 三因子模型演示")
    print("=" * 60)

    # 1. 生成模拟数据
    print(f"\n[1/4] 生成 {N_STOCKS} 只股票 × {N_DAYS} 天的模拟数据...")
    data = simulate_stock_universe(N_STOCKS, N_DAYS)
    print(f"      数据量: {len(data)} 行, "
          f"日期范围: {data['date'].min().date()} ~ {data['date'].max().date()}")

    # 2. 构建因子
    print("\n[2/4] 构建 MKT / SMB / HML 因子...")
    factors = build_factors(data)
    print(f"      因子数据: {len(factors)} 天")
    print(f"      MKT 日均: {factors['MKT'].mean():.4%},  波动率(年): {factors['MKT'].std() * np.sqrt(252):.1%}")
    print(f"      SMB 日均: {factors['SMB'].mean():.4%},  波动率(年): {factors['SMB'].std() * np.sqrt(252):.1%}")
    print(f"      HML 日均: {factors['HML'].mean():.4%},  波动率(年): {factors['HML'].std() * np.sqrt(252):.1%}")

    # 3. 因子暴露（选 5 只示例股票）
    print("\n[3/4] 计算因子暴露（OLS 回归）...")
    sample_codes = [f"{600000 + i:06d}" for i in range(0, 30, 6)]
    print(f"\n{'股票':<8} {'α(日)':>8} {'β_mkt':>7} {'β_smb':>7} {'β_hml':>7} {'R²':>7}  {'特征解读'}")
    print("-" * 80)

    for code in sample_codes:
        group = data[data["stock_code"] == code]
        ret_series = group.set_index("date")["daily_ret"]
        exp = calc_exposure(ret_series, factors)

        # 解读
        tags = []
        if exp["beta_smb"] > 0.3:
            tags.append("小盘偏多")
        elif exp["beta_smb"] < -0.3:
            tags.append("大盘偏多")
        else:
            tags.append("规模中性")

        if exp["beta_hml"] > 0.3:
            tags.append("价值偏多")
        elif exp["beta_hml"] < -0.3:
            tags.append("成长偏多")
        else:
            tags.append("价值中性")

        print(f"{code:<8} "
              f"{exp['alpha_daily']:>+8.4%} "
              f"{exp['beta_mkt']:>7.3f} "
              f"{exp['beta_smb']:>7.3f} "
              f"{exp['beta_hml']:>7.3f} "
              f"{exp['r2']:>7.3f}  "
              f"{', '.join(tags)}")

    # 4. 画图 & 横截面验证
    print("\n[4/4] 可视化 & 横截面验证...")
    plot_factors(factors, OUTPUT_DIR / "factor_cumulative_returns.png")
    cross_section_check(data, factors)

    print(f"\n{'='*60}")
    print("完成。关键小结：")
    print("  1. 因子模型 = 用少数几个可量化特征解释大量股票的收益")
    print("  2. 因子暴露 = 回归系数，描述股票对因子的敏感度")
    print("  3. 因子收益 = 因子组合的收益差，代表这个因子当期的'回报'")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
