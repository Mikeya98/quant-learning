"""ST / *ST 股票识别。

A 股主板 ST 股的涨跌停幅度是 ±5%，创业板/科创板 ST 仍是 ±20%，
北交所 ±30%。回测里如果不知道历史 ST 状态，至少可以用当前股票
名称做一层粗筛：名称以 "ST"、"*ST"、"S*ST" 开头。
"""

import re
from typing import Dict, Iterable, Set

_ST_RE = re.compile(r"^\s*S?\*?ST", re.IGNORECASE)


def is_st_name(name: str) -> bool:
    """股票名称是否带 ST / *ST 前缀。"""
    return bool(_ST_RE.match(str(name or "")))


def st_symbols_from_names(names: Dict[str, str]) -> Set[str]:
    """从 {代码: 名称} 里挑出 ST 代码集合。"""
    return {str(sym) for sym, name in names.items() if is_st_name(name)}


def st_symbols_from_file(path: str) -> Set[str]:
    """从一行一个代码的文本文件读取 ST 名单（空行和 # 注释忽略）。"""
    out: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.add(line.split(",")[0].strip())
    return out
