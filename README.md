# quant-learning

写嵌入式 C 写了好几年，最近突然想搞清楚量化交易到底是怎么回事。

市面上回测框架一堆，装一个就能跑，但跑完不知道那堆数字怎么来的。所以决定从零写一个——只依赖 pandas 和 matplotlib，不用任何量化库。

代码在写，文章也在写。每个阶段跑通一个模块，再写一篇总结。

## 目录

```
quant_learning/
├── main.py           # 命令行入口
├── src/
│   ├── data/         # 数据获取和清洗（akshare → CSV）
│   ├── strategy/     # 策略实现（基类 + 具体策略）
│   ├── engine/       # 回测引擎（待完成）
│   ├── analysis/     # 绩效分析（待完成）
│   └── risk/         # 风险管理（待完成）
├── zhihu/            # 知乎系列文章草稿
└── data/             # 本地缓存（gitignore）
```

## 进展

- Phase 1: 数据层。akshare 下载 A 股日线，清洗/停牌处理/前复权
- Phase 2: 策略层。双均线、MACD、RSI、布林带，用基类统一接口
- Phase 3: 回测引擎（待开始）

## 跑起来

```bash
pip install -r requirements.txt
python main.py fetch 000001      # 下载股票数据
python main.py signal ma_cross 000001   # 看策略信号
```

## License

MIT
