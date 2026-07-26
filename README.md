# Quant Learning — 从零搭建 A 股回测框架

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个用于**学习量化交易**的迷你回测框架。**不调包、不照搬**，每行代码都从零写起。

## 为什么造轮子？

现成的回测框架（backtrader、vnpy、zipline）能直接跑策略，但你在用黑盒。这个项目的目标是：

> **搞懂回测的每一行代码在做什么，而不是调个 `run()` 函数**

## 目录结构

```
quant_learning/
├── main.py                  # CLI 入口
├── src/
│   ├── data/                # 数据层：A股获取/清洗/存储
│   ├── strategy/            # 策略层：基类 + 经典策略
│   ├── engine/              # 引擎层：回测主循环/撮合/持仓
│   ├── analysis/            # 分析层：绩效指标/可视化
│   └── risk/                # 风险层：仓位管理/止损
├── data/                    # CSV 数据缓存
├── notebooks/               # Jupyter 探索笔记
└── strategies/              # 策略参数配置
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载数据
python main.py fetch 000001

# 查看数据概况
python main.py info 000001

# 列出已下载的股票
python main.py list
```

## 学习路线

| Phase | 内容 | 状态 |
|-------|------|------|
| 1 | 数据基础：A股获取/清洗/存储 | ✅ 完成 |
| 2 | 策略框架：基类 + 4个经典策略 | 🚧 进行中 |
| 3 | 回测引擎：主循环/撮合/持仓跟踪 | 📋 计划中 |
| 4 | 绩效分析：夏普比率/最大回撤/可视化 | 📋 计划中 |
| 5 | 风险管理：仓位管理/止损/参数优化 | 📋 计划中 |
| 6 | 多策略对比 + 完整报告 | 📋 计划中 |

## 知乎系列（同步发布）

- [ ] 第1篇：为什么从零写回测框架
- [ ] 第2篇：用 Python 获取 A 股数据
- [ ] 第3篇：4 个经典策略，20 行一个
- [ ] 第4篇：手写回测引擎的真实成本
- [ ] 第5篇：为什么最佳参数实盘亏钱
- [ ] 第6篇：从零到完整系统的复盘

## License

MIT
