---
name: tick-downsample
description: 为 Orion 构建、开发、测试或校验从 CTP 等原始高频 Tick 快照流降频聚合为 1 分钟 Bar 基础行情与微观结构因子的算子管道。严格遵循纯 Polars Lazy 链式范式与 docs/tick_downsample_1min_features.md 规范，支持 OHLCV 基础字段及资金流、订单流 OFI、盘口深度失衡、期货持仓多空博弈、日内均价与涨跌停锚点特征。用户在要求编写 Tick 降频算子、微观底表特征提取、重采样聚合管道时使用。
---

# Tick 降频与微观特征开发 (Tick Downsampling to 1-Minute Bar)

本技能专门用于从 **CTP 500ms 原始快照流** 降频聚合生成 **1 分钟 Bar 基础行情与全套微观结构因子**。

实现严格遵循 [docs/tick_downsample_1min_features.md](../../docs/tick_downsample_1min_features.md) 与 [references/tick-contract.md](references/tick-contract.md)，采用纯 Polars Lazy 链式表达，确保极高的计算吞吐量与内存友好性。

---

## 必须遵循的开发流程

1. **阅读输入规范与契约**：
   - 完整阅读 [references/tick-contract.md](references/tick-contract.md) 与 [docs/tick_downsample_1min_features.md](../../docs/tick_downsample_1min_features.md)。
   - 确认输入数据严格匹配 CTP 原生 15 个基础字段（`TradingDay, InstrumentID, UpdateTime, UpdateMillisec, LastPrice, Volume, Turnover, AveragePrice, BidPrice1, BidVolume1, AskPrice1, AskVolume1, OpenInterest, UpperLimitPrice, LowerLimitPrice`）。
2. **复用或扩展模板**：
   - 快速单类别特征抽取：以 [assets/resampler_template.py](assets/resampler_template.py) 为起点编写专属提取器；
   - 完整全量特征宽表计算：直接调用或扩展 [assets/full_resampler.py](assets/full_resampler.py)；
   - 外部数据加载：复用 [assets/tick_dataloader.py](assets/tick_dataloader.py)。
3. **四阶段链式 Polars Lazy 实现**：
   - **阶段 1：时间戳构建与排序**：构建标准 `timestamp`，按 `['InstrumentID', 'timestamp']` 严格升序；
   - **阶段 2：Tick 级增量与打标**：必须配合 `.over('InstrumentID')` 进行全局差分（$\Delta V, \Delta M, \Delta OI$）、Lee-Ready 交易方向 $D_t \in \{+1, -1, 0\}$ 判定与 Cont 一档 OFI 递推；
   - **阶段 3：1 分钟时间切片与分组聚合**：统一使用 `dt.truncate('1m')` 或 `group_by_dynamic('timestamp', every='1m', closed='right')` 聚合；
   - **阶段 4：Bar 级比率归一化与除零保护**：优先产出 **🟢 绿灯无量纲特征**（除法必须加小量 $\epsilon = 10^{-7}$ 或使用 `safe_div` 保护）。
4. **运行静态合规校验**：
   - 运行 `python3 skills/tick-downsample/scripts/validate_tick_resampler.py path/to/resampler.py`；
   - 运行 `python3 -m py_compile path/to/resampler.py`。
5. **最小测试用例验证**：
   - 使用内存模拟的最小 `pl.LazyFrame` 执行 `.collect()` 测试，校验列完整性、主键唯一性 `['trade_time', 'code']` 以及数值合理性（无意外 NaN/inf）。

---

## 核心架构原则与不可变更契约

- **纯函数式与纯 Lazy 机制**：
  - 算子模块内部严禁触发物理执行或写出（严禁 `.collect()`、`.fetch()`、`sink_*`、`to_pandas`）；
  - 严禁 Python 逐行遍历（`iter_rows`）或自定义 Python 函数（UDF），所有计算必须由 Polars 原生表达式实现。
- **全物理真实价格计算（0 复权）**：
  - 降频算子内严格使用原始物理 Tick 数据计算，**严禁做任何形式的主力复权**；
  - 确保盘口撮合自洽性与持仓量增减真实性。
- **主键契约与输出列**：
  - 统一返回 `pl.LazyFrame`；
  - 结果主键必须为 `trade_time`（分钟时间戳）与 `code`（合约代码）；
  - 移除所有以下划线开头的单跳临时列（如 `_delta_v`），保持最终宽表整洁。
- **与下游因子开发技能的对接**：
  - 降频算子生成的 1 分钟宽表（基础量价与微观指标），直接作为下游 [skills/factor-development](../factor-development/SKILL.md) 的输入底表；
  - 下游基于此底表进一步开发多周期滚动因子（如 `rolling_rank(ofi_normalized, 250)`）。

---

## 八大分类特征快速索引

| 特征分类 | 核心代表特征 | 适用算子位置 |
| :--- | :--- | :--- |
| **`future_1min`** | `open, high, low, close, volume, money, vwap, volume_per_tick` | 基础 OHLCV 与单笔粗细 |
| **`money_flow`** | `tick_in_pct, volume_in_pct, net_money_in_pct, smart_net_vol_pct` | Lee-Ready 资金流与聪明钱 |
| **`imbalance`** | `bid_ask_spread, depth_imbalance_1, micro_price_bias, realized_volatility, jump_ratio` | 盘口深度失衡与波动跳跃分解 |
| **`order_flow`** | `ofi_normalized, voi_normalized, depletion_imbalance` | Cont 一档订单流与防线击穿 |
| **`corr`** | `corr_money_ret, corr_ofi_ret, corr_vol_depth_imb` | 量价/订单流微观协方差 |
| **`open_interest`**| `delta_oi_ratio, double_open_ratio, bull_active_open_ratio, oi_flow_imbalance` | 期货特有持仓博弈四象限 |
| **`average_price_anchor`** | `price_to_avg_dev, vwap_to_avg_bias, price_above_avg_time` | 当日结算均价偏离与多头控盘 |
| **`limit_bounds`** | `dist_upper_limit, dist_lower_limit, limit_bound_asymmetry` | 距涨跌停空间与流动性边界 |
