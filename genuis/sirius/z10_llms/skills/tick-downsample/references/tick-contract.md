# Orion Tick 降频到 1 分钟 Bar 特征契约

本契约依据 [docs/tick_downsample_1min_features.md](../../../docs/tick_downsample_1min_features.md) 制定，是使用纯 Polars Lazy 范式将原始 500ms CTP Tick 快照降频聚合为 1 分钟 Bar 基础字段及微观结构因子的实现规范。

---

## 1. 输入数据契约 (Input Schema)

降频算子接收外部注入的 `df_lazy: pl.LazyFrame`，输入数据必须至少包含以下 15 个 CTP 原生字段（不得随意拼写、更名或增删）：

| 字段名称 | 推荐类型 | 说明 |
| :--- | :--- | :--- |
| `TradingDay` | `pl.Utf8` / `pl.Int64` | 交易日（YYYYMMDD） |
| `InstrumentID` | `pl.Utf8` | 合约代码（如 `rb2410`, `IF2409`） |
| `UpdateTime` | `pl.Utf8` | 更新时间（HH:MM:SS） |
| `UpdateMillisec` | `pl.Int64` | 毫秒（0 或 500） |
| `LastPrice` | `pl.Float64` | 最新成交价 |
| `Volume` | `pl.Int64` / `pl.Float64` | 当日累计成交量（以手为单位） |
| `Turnover` | `pl.Float64` | 当日累计成交金额（元） |
| `AveragePrice` | `pl.Float64` | 当日累计均价（$\frac{Turnover}{Volume \times Multiplier}$） |
| `BidPrice1` | `pl.Float64` | 买一价 |
| `BidVolume1` | `pl.Int64` | 买一量 |
| `AskPrice1` | `pl.Float64` | 卖一价 |
| `AskVolume1` | `pl.Int64` | 卖一量 |
| `OpenInterest` | `pl.Float64` / `pl.Int64` | 当日当前持仓量（未平仓合约数） |
| `UpperLimitPrice` | `pl.Float64` | 涨停板价 |
| `LowerLimitPrice` | `pl.Float64` | 跌停板价 |

> **时间戳构造规则**：
> 进入聚合前，需将 `TradingDay`、`UpdateTime`、`UpdateMillisec` 组合成标准时间戳列 `timestamp: pl.Datetime`。若外部数据已提供标准 `timestamp` 列，优先直接复用。

---

## 2. 四阶段链式 Polars Lazy 流水线标准

为了兼顾计算精度、向量化性能与内存占用，Tick 降频必须严格遵循以下 **四个分层阶段**：

```mermaid
flowchart LR
    Stage1[阶段 1: 时间戳构建与排序] --> Stage2[阶段 2: Tick 级增量与微观打标]
    Stage2 --> Stage3[阶段 3: 1分钟时间切片与分组聚合]
    Stage3 --> Stage4[阶段 4: Bar 级比率归一化与除零保护]
```

### 阶段 1：时间戳构建与标的排序
- 确保按 `['InstrumentID', 'timestamp']` 严格升序排序。
- 保证每个品种的时序是连续且独立的。

### 阶段 2：Tick 级增量与微观打标 (Tick Primitives)
在聚合之前，在 Tick 流上完成以下向量化表达式计算（必须使用 `.over('InstrumentID')`）：
1. **基础单跳差分**：
   - `delta_v = pl.col('Volume').diff().clip(lower_bound=0).over('InstrumentID')`
   - `delta_m = pl.col('Turnover').diff().clip(lower_bound=0).over('InstrumentID')`
   - `delta_oi = pl.col('OpenInterest').diff().over('InstrumentID')`
   - 首个 Tick 边界处理：空值填充为 0 或单跳累积量。
2. **中间价与微观价**：
   - `mid_price = (pl.col('AskPrice1') + pl.col('BidPrice1')) / 2.0`
   - `spread = pl.col('AskPrice1') - pl.col('BidPrice1')`
   - `micro_price = (pl.col('AskPrice1') * pl.col('BidVolume1') + pl.col('BidPrice1') * pl.col('AskVolume1')) / (pl.col('BidVolume1') + pl.col('AskVolume1') + 1e-7)`
3. **主动买卖方向 (Lee-Ready 规则)**：
   - 比较 $LastPrice_t$ 与上一跳报价 $AskPrice1_{t-1}, BidPrice1_{t-1}$ 及上一跳价格 $LastPrice_{t-1}$，输出 $D_t \in \{+1, -1, 0\}$。
4. **一档 Cont (2014) OFI 增量**：
   - 计算 $I_{bid, t}$ 与 $I_{ask, t}$，输出单跳 $OFI_t = I_{bid, t} - I_{ask, t}$。
5. **博弈属性打标**：
   - 双开、双平、主动多头增仓、主动空头增仓、多头砍仓、空头逼仓的单跳布尔掩码。

### 阶段 3：1 分钟时间切片与分组聚合 (Aggregation)
将 Tick 数据降频切片聚合为 1 分钟 Bar：
- **切片规则**：基于 `timestamp` 截断为分钟：`bar_time = pl.col('timestamp').dt.truncate('1m')`，或使用 `group_by_dynamic('timestamp', every='1m', closed='right')`。
- **分组键**：统一按 `['bar_time', 'InstrumentID']`（或对齐为 `['trade_time', 'code']`）。
- **聚合算子集**：
  - 极值与端点：`pl.col('LastPrice').first().alias('open')`，`max().alias('high')`，`min().alias('low')`，`last().alias('close')`
  - 累积求和：`delta_v.sum().alias('volume')`，`delta_m.sum().alias('money')`，`ofi.sum().alias('ofi_sum')`
  - 样本密度：`pl.len().alias('tick_count')`
  - 条件求和：`delta_v.filter(is_buyer).sum().alias('volume_in')`，`delta_v.filter(is_double_open).sum().alias('double_open_vol')`
  - 波动率统计：`pl.col('log_ret').drop_nulls().std().alias('realized_volatility')`
  - 微观协方差：`pl.corr('delta_m', 'log_ret').alias('corr_money_ret')`

### 阶段 4：Bar 级比率归一化与除零保护 (Normalization)
聚合产出的宽表进入最后的比率化阶段：
- 所有分母除法必须使用 `safe_div(numerator, denominator, fill_value=0.0)` 或 `+ 1e-7` 保护，防止零成交分钟或停板时产生 `NaN` / `inf`。
- 生成规范中重点推荐的 **🟢 绿灯无量纲特征**：
  - `volume_in_pct = volume_in / (volume + 1e-7)`
  - `ofi_normalized = ofi_sum / (volume + 1e-7)`
  - `delta_oi_ratio = delta_oi / (volume + 1e-7)`
  - `double_open_ratio = double_open_vol / (volume + 1e-7)`
  - `volume_per_tick = volume / (tick_count + 1e-7)`

---

## 3. 输出数据契约 (Output Schema)

1. **结构要求**：
   - 统一返回 `pl.LazyFrame`；
   - 主键列必须为 `['trade_time', 'code']`（时间戳与合约代码）；
   - 后续列为生成的具体特征列（基础量价或微观因子）；
   - 禁止泄漏阶段性生成的单跳中间临时列（如 `_delta_v`, `_is_buyer` 等）。
2. **唯一性**：每个 `(trade_time, code)` 在输出表中必须唯一。

---

## 4. Polars 编码纪律与性能规范

1. **全链路保持 Lazy 状态**：
   - 算子模块内部严禁出现任何触发计算或写出的调用：禁止 `.collect()`、`.fetch()`、`.sink_*()`、`.to_pandas()`、`.to_numpy()`。
   - 最终物理执行必须交由外部调度器或数据流水线触发。
2. **严禁 Python 逐行循环**：
   - 严禁用 `for row in df.iter_rows()`、`apply()` 或自定义 Python UDF 遍历 Tick 数据；
   - 所有的 Lee-Ready 判定、OFI 计算、持仓状态分类必须 100% 使用 Polars 原生表达式（`pl.when(...).then(...).otherwise(...)`）完成 SIMD/多线程向量化加速。
3. **零成交 Tick 的分别处理**：
   - 当单跳 $\Delta Volume_t = 0$ 时，该跳不属于实际撮合成交；
   - 资金流统计（`volume_in`, `vwap_tick` 等）在表达式中必须过滤 `delta_v > 0`；
   - 盘口深度失衡（`depth_imbalance`）、价差（`spread`）与订单流（`ofi`）必须包含所有 Tick 快照。
4. **合约乘数（Multiplier）适配**：
   - 算子应支持传入合约乘数映射表 `multipliers: dict[str, float] | None`；
   - 当计算单笔真实均价或资金金额时，使用对应的品种乘数进行换算。
