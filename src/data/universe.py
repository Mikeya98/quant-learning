"""
股票池 —— 从全市场里挑出要研究的一批股票。

因子研究对样本量的要求比策略回测高得多：8 只股票算出来的 IC
标准误大概是几百只股票的好几倍，几乎什么都说明不了。所以这里
做一个最小可用的"选池"流程：

    1. 拉全市场快照（东方财富 push2 接口），拿代码 / 名称 / 市值
    2. 过滤：ST、退市、北交所、上市不足 N 天（用代码前缀先粗筛）
    3. 按流通市值排序，取前 N 只 —— 大市值股票流动性好，回测里
       的成交假设也更接近真实
    4. 逐只补行业分类（stock/get 的 f127 字段）
    5. 结果落到 data/meta/universe.csv，后面的命令直接读这个文件

接口限流很常见（尤其是连续请求之后），所有网络调用都带重试和
退避；失败时抛出异常，由调用方决定是重试还是退回本地缓存。

解析部分和网络部分是分开的：`parse_snapshot` / `parse_industry`
只吃 dict，可以离线用样例数据测试。
"""

import os
import time
from typing import Iterable, Optional

import pandas as pd
import requests

from .meta import META_DIR

UNIVERSE_FILE = os.path.join(META_DIR, "universe.csv")

SNAPSHOT_URL = "https://push2.eastmoney.com/api/qt/clist/get"
STOCK_INFO_URL = "https://push2.eastmoney.com/api/qt/stock/get"

# 沪深 A 股: 深主板 + 创业板 + 沪主板 + 科创板
A_SHARE_FILTER = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Referer": "https://quote.eastmoney.com/",
}


# ===================================================================
# 解析（纯函数，可离线测试）
# ===================================================================

def parse_snapshot(payload: dict) -> pd.DataFrame:
    """
    把 clist 接口返回的 JSON 解析成表。

    字段含义（东方财富自己的编号）:
        f12 代码   f14 名称   f20 总市值   f21 流通市值
    """
    data = payload.get("data") or {}
    diff = data.get("diff") or []

    rows = []
    for item in diff:
        code = str(item.get("f12", "")).zfill(6)
        rows.append({
            "symbol": code,
            "name": item.get("f14", ""),
            "total_mktcap": item.get("f20"),
            "float_mktcap": item.get("f21"),
        })

    df = pd.DataFrame(rows, columns=["symbol", "name",
                                     "total_mktcap", "float_mktcap"])
    return df


def filter_universe(df: pd.DataFrame, exclude_st: bool = True) -> pd.DataFrame:
    """去掉 ST / 退市 / 北交所。

    有流通市值时按流通市值降序; 只有成交额时 (东财快照被限流的降级
    路径) 按当日成交额降序 —— 流动性好的股票同样适合做组合回测。
    """
    out = df.copy()

    if exclude_st:
        name = out["name"].fillna("")
        out = out[~name.str.contains("ST|退", na=False)]

    # 北交所代码 8 开头、4 开头
    out = out[~out["symbol"].str.startswith(("4", "8"))]

    if "float_mktcap" not in out.columns:
        out["float_mktcap"] = pd.NA

    if out["float_mktcap"].notna().any():
        out = out.dropna(subset=["float_mktcap"])
        out = out.sort_values("float_mktcap", ascending=False)
    elif "amount" in out.columns:
        out = out.dropna(subset=["amount"])
        out = out.sort_values("amount", ascending=False)
    else:
        out = out.dropna(subset=["float_mktcap"])

    return out.reset_index(drop=True)


def parse_industry(payload: dict) -> Optional[str]:
    """从 stock/get 的返回里取行业名 (f127)"""
    data = payload.get("data") or {}
    ind = data.get("f127")
    return str(ind) if ind else None


# ===================================================================
# 网络
# ===================================================================

def _get_json(url: str, params: dict, retries: int = 3,
              backoff: float = 1.5, timeout: int = 15) -> dict:
    """带重试的 GET。失败会抛最后一次的异常。"""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:          # noqa: BLE001 - 网络异常种类太多
            last = exc
            if attempt + 1 < retries:
                time.sleep(backoff ** attempt)
    raise RuntimeError(f"请求失败: {url} ({last})")


def fetch_snapshot(page_size: int = 100, max_pages: int = 60) -> pd.DataFrame:
    """
    拉全市场快照。

    分页拉取，直到返回为空或者达到 max_pages。6000 只股票在
    page_size=100 时需要 60 页，接口本身支持更大的 pz，但太大
    容易被限流。
    """
    frames = []
    for page in range(1, max_pages + 1):
        payload = _get_json(SNAPSHOT_URL, {
            "pn": page,
            "pz": page_size,
            "po": 1,
            "np": 1,
            "fltt": 2,
            "invt": 2,
            "fid": "f21",                 # 按流通市值排序
            "fs": A_SHARE_FILTER,
            "fields": "f12,f14,f20,f21",
        })
        df = parse_snapshot(payload)
        if df.empty:
            break
        frames.append(df)

        total = (payload.get("data") or {}).get("total") or 0
        if page * page_size >= total:
            break

    if not frames:
        return parse_snapshot({})
    return pd.concat(frames, ignore_index=True)


def fetch_sina_snapshot(progress=print) -> pd.DataFrame:
    """降级快照: 新浪全市场行情。

    东财 clist 接口限流时用它兜底。新浪没有流通市值字段, 用当日
    成交额做流动性排序; 后续补行业仍然走东财 stock/get (两个接口
    独立限流, 通常一个被封另一个还能用)。
    """
    import akshare as ak

    progress("[universe] 东财快照不可用，降级到新浪 spot ...")
    df = ak.stock_zh_a_spot()
    if df is None or len(df) == 0:
        raise RuntimeError("新浪快照也为空")

    work = df.copy()
    work["symbol"] = (
        work["代码"].astype(str).str.replace(r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
    )
    work["name"] = work["名称"].astype(str)
    work["amount"] = pd.to_numeric(work.get("成交额"), errors="coerce")
    work["float_mktcap"] = pd.NA
    return work[["symbol", "name", "float_mktcap", "amount"]]


def fetch_industry(symbol: str, retries: int = 2) -> Optional[str]:
    """单只股票的行业（东方财富的行业分类，约 86 个二级行业）"""
    symbol = str(symbol).zfill(6)
    market = "1" if symbol[0] in "69" else "0"
    payload = _get_json(STOCK_INFO_URL, {
        "secid": f"{market}.{symbol}",
        "fields": "f57,f58,f127",
    }, retries=retries)
    return parse_industry(payload)


# ===================================================================
# 组合流程
# ===================================================================

def build_universe(top_n: int = 50, exclude_st: bool = True,
                   with_industry: bool = True,
                   progress=print) -> pd.DataFrame:
    """
    选池主流程：快照 → 过滤 → 取前 N → 补行业。

    Returns
    -------
    DataFrame: symbol, name, industry, float_mktcap
    """
    try:
        snapshot = fetch_snapshot()
    except Exception as exc:          # noqa: BLE001
        progress(f"[universe] 东财快照失败: {exc}")
        snapshot = fetch_sina_snapshot(progress=progress)
    if snapshot.empty:
        raise RuntimeError("快照为空，接口可能被限流了")

    pool = filter_universe(snapshot, exclude_st=exclude_st).head(top_n)
    cols = ["symbol", "name", "float_mktcap"]
    if "amount" in pool.columns:
        cols.append("amount")
    pool = pool[cols].copy()
    pool["industry"] = None
    if pool["float_mktcap"].notna().any():
        pool["rank_by"] = "float_mktcap"
    else:
        pool["rank_by"] = "amount"

    if with_industry:
        for i, sym in enumerate(pool["symbol"]):
            try:
                pool.loc[i, "industry"] = fetch_industry(sym)
            except Exception as exc:      # noqa: BLE001
                progress(f"  [WARN] {sym} 行业获取失败: {exc}")
            if (i + 1) % 10 == 0:
                progress(f"  行业进度 {i + 1}/{len(pool)}")

    return pool


def save_universe(pool: pd.DataFrame, path: str = None) -> str:
    path = path or UNIVERSE_FILE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pool.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_universe(path: str = None) -> pd.DataFrame:
    path = path or UNIVERSE_FILE
    if not os.path.exists(path):
        return pd.DataFrame(columns=["symbol", "name", "industry", "float_mktcap"])
    df = pd.read_csv(path, dtype={"symbol": str}, encoding="utf-8-sig")
    df["symbol"] = df["symbol"].str.zfill(6)
    return df
