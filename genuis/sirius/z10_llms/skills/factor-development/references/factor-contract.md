# Orion 因子模块契约

本契约提炼自 `demo/xy_factor_history/`，是编写 Orion 因子时的格式依据。

参考目录中的公式与编码风格可以复用，但其中由因子自行调用 `dataloader`、接收日期参数和执行 `.collect()` 的结构已被外部数据注入契约取代。

## 标准文件结构

每种因子逻辑都是独立的 Python 模块：

1. 与正式 `feature/` 目录的成熟实现一致，模块文档字符串使用 `"""因子定义: ..."""`，简洁说明核心公式。因子名、批次、数据粒度、必需字段和 `max_window` 由外层生成报告及注册信息维护，避免代码文档与机器产物形成两套元数据。
2. `import polars as pl`
3. 参数化因子在导入之后定义模块级大写默认参数常量。
4. `calculate(...)` 实现单个参数对应的因子公式。简单公式可返回 `pl.Expr`；复杂分阶段公式应接收 `df_lazy: pl.LazyFrame` 并返回 `pl.LazyFrame`。不得负责参数集合循环。
5. `compute(df_lazy, ...) -> pl.LazyFrame` 作为外部入口；附加参数提供模块顶部常量作为默认值，负责参数编排并调用 `calculate`。
6. 中文函数文档说明输入 LazyFrame、参数、必需字段和动态或固定返回列。

以 [../assets/factor_template.py](../assets/factor_template.py) 为起点。

跨因子共享的安全除法、对数收益、参数校验和批量注册表计算统一使用 `feature/utils/common.py`。具体因子模块按需绝对导入，例如：

```python
from feature.utils.common import log_return, safe_div, validate_period
```

不得在因子分类目录中创建 `_common.py` 或复制同类公共函数；只有确实只服务单个公式的辅助表达式才保留在该因子文件内。

因子公式必须直接由原生 Polars 表达式和 LazyFrame 阶段组成，包括 `pl.col(...)`、`pl.when(...)`、`.with_columns(...)` 和 `.over("code")`。`feature/utils/formula.py` 维护供因子挖掘、表达式枚举和实验搜索使用的原生 `pl.Expr` 原子算子，但正式因子模块不得依赖该挖掘层；确定公式后必须展开为显式的原生 Polars 阶段。

标准复杂因子必须采用链式 LazyFrame 分层结构：每个 `.with_columns(...)` 只计算当前依赖层，后续阶段仅通过 `pl.col("_临时列")` 使用上一层结果。简单、互不依赖的表达式可以位于同一个阶段；一旦表达式包含 `shift`、`diff`、`pct_change`、`rolling_*`、`ewm_*` 或累计运算，并且该结果还要参与另一个时序或聚合运算，就必须先设置明确别名并在下一阶段计算。禁止 `rolling_corr(pl.col("x"), pl.col("y").shift(1).over("code"), ...)` 这类嵌套写法。

## 分类批次目录与因子名称

批次目录采用 `<因子家族><市场范围><三位批次号>`：

- 家族 `t` 表示技术与量价因子，`m` 表示 Tick、逐笔成交、订单簿和订单流等微观结构因子。
- 市场范围 `c` 表示期现通用，`f` 表示期货专属，`s` 表示现货专属，`b` 表示期现联合。
- `tc001` 表示通用技术量价因子的第一批目录；它不是因子名称或输出列名称。
- 三位数字只用于拆分同一分类下的代码批次，不表达公式、参数、频率、优先级或版本。

目录直接使用：

```text
feature/tc001/
feature/tf001/
feature/ts001/
feature/tb001/
feature/mc001/
feature/mf001/
feature/ms001/
feature/mb001/
```

每个目录包含多个小写 snake_case 因子文件，例如 `feature/tc001/mom.py`。默认使用说明书的语义名称；显式传入 `--factor-name` 时允许调用方编号（例如 `tf001.py`），并要求文件名、元数据名和输出列保持一致。批次目录仍必须遵守独立的 `<家族><市场范围><三位批次号>` 规则。

分类按实际公式依赖判断：只使用通用 OHLCV/value 字段的公式进入 `tcXXX`；使用 `openint`、合约结构或其他期货专属语义才进入 `tfXXX`；同时依赖 `future_*` 和 `spot_*` 的公式进入 `tbXXX`。原始因子名称不能代替字段依赖判断。

数据粒度登记为 `bar`、`tick`、`trade`、`orderbook` 或 `mixed`。具体分钟、小时或日周期不进入批次目录或因子名称规则。

## 包导出与全局注册表

- 批次目录的 `__init__.py` 只负责显式导出，每个实现文件对应一条 `from .mom import compute as mom_compute`。子包不得维护注册表或批量入口。
- 唯一全局注册位置是 `feature/__init__.py`。它维护 `FACTOR_SPECS`、`FACTORS`、`BATCH_FACTORS`、`FACTOR_BATCHES`、家族索引、市场范围索引和数据粒度索引。
- 每个因子注册一个 `FactorSpec`，记录语义名称、家族、市场范围、数据粒度、所属批次、必需字段和 `compute`。不同索引引用同一个实现对象，不得复制代码。
- 顶层 `compute_batch(df_lazy)` 默认计算 `COMMON_FACTORS`。其他类别通过 `names` 显式选择。
- 外部统一从 `feature` 导入注册表、元数据和批量入口。

## 命名与字段结构

- 因子文件使用小写 snake_case 语义名称；一个 `.py` 文件只包含一种因子逻辑。批次目录名必须匹配 `^(t|m)(c|f|s|b)\d{3}$`。
- 无参数因子仍要求文件名主干、`因子名称`、最终 `.alias(...)` 和第三个返回列一致。
- 同一公式仅参数不同的因子合并到一个参数化文件。文件名和元数据使用语义因子名，输出列使用 `<因子名>_<参数>`；例如 `mom.py` 接收 `windows=(5, 10)` 并输出 `mom_5`、`mom_10`。
- 普通行情公共字段固定为：`trade_time`、`code`、`high`、`low`、`open`、`close`、`volume`、`value`。期货行情额外要求 `openint`，现货行情不要求该字段。
- 普通因子只能引用上述字段。不得根据旧示例使用 `date`、`Code`、`closePrice` 等名称，也不得自行猜测或创建新的原始输入字段。用户确认新增基础字段后，应先更新本清单和校验器。
- 允许通过 `.alias(...)` 创建公式所需的临时计算列和最终因子列，但临时列不得进入最终结果。
- 无参数因子精确返回 `['trade_time', 'code', '<factor_name>']`；参数化因子返回 `['trade_time', 'code', *factor_columns]`。两者均不得泄漏辅助列。
- 除非公式明确改变品种范围，否则每个 `(trade_time, code)` 保持一行；连接操作不得导致键值行数膨胀。

## 旧字段的统一派生规则

旧因子中的以下字段不得作为外部输入字段，必须在因子内部从已登记的 `close` 确定性派生：

- `preClosePrice`：`pl.col('close').shift(1).over('code')`
- `chgPct`：`pl.col('close').pct_change().over('code')`

派生前必须先按 `['trade_time', 'code']` 排序，且必须按 `code` 隔离序列。可以为表达式设置语义清晰的临时别名，但不得创建名为 `preClosePrice` 或 `chgPct` 的新原始输入约定，也不得把临时列带入最终结果。

`turnoverRate` 当前不属于基础字段，也没有获准的统一派生公式。迁移旧因子时，凡公式依赖 `turnoverRate` 的因子暂时跳过并报告；不得用 `volume`、`value` 或其他字段自行替代。

`vwap` 也不是外部基础字段。普通行情需要 VWAP 时统一使用 `value / volume` 派生；`volume` 为零或空值时返回空值，不得使用任意常数填充分母。基差输入分别使用 `future_value / future_volume` 与 `spot_value / spot_volume`，并采用相同的零分母处理。派生 VWAP 仅作为临时列，不得扩展基础字段契约或泄漏到最终输出。

## 输入数据与窗口

数据加载必须在因子模块外完成。外部调度器读取元数据中的 `数据来源` 和 `max_window`，加载足量数据后，将准备好的 `pl.LazyFrame` 直接传入 `compute`。

输入 Schema 校验也由外部调度器和运行验证负责。正式因子模块不得读取 `df_lazy.columns`、`df_lazy.schema` 或调用 `collect_schema()`；这些操作会触发 Schema 解析并把输入准备职责重新带回因子模块。

多数据源场景由外部调度器分别加载、裁剪和连接数据，再把连接后的单个 LazyFrame 传给因子。因子模块不得负责数据源选择或连接，不得导入或调用 `dataloader`，也不得接收 `begDate`、`endDate` 等加载参数。

外部调用示意：

```python
from file_dataloader import dataloader

df_lazy = dataloader('/data/recent_data.feather')
factor_df = (
    factor_module.compute(df_lazy)
    .collect()
)
```

外部代码可以复用同一个 LazyFrame 为多个因子供数。多数据源因子必须先在外部完成连接，以保持因子接口统一。

## 参数化因子

- Python 流水线每次只生成说明书中的一个因子和一组默认参数，不在同一文件展开参数网格或生成多个输出列。
- 默认参数放在导入语句之后、函数之前，分别命名为 `DEFAULT_PERIOD`、`DEFAULT_FAST`、`DEFAULT_THRESHOLD` 等大写常量。优先使用独立、带类型的参数，不把全部参数塞进一个无类型 `params` 元组。
- 复杂分阶段公式使用成熟 `feature/` 风格：`calculate(df_lazy: pl.LazyFrame, ...) -> pl.LazyFrame` 完成计算图并可直接精确选择最终三列；`compute` 负责参数检查、排序和调用 `calculate`。若 `calculate` 已经最终选列，`compute` 不必重复 `.select(...)`。
- 只有最终公式确实适合一个表达式时，才使用 `calculate(...) -> pl.Expr`，由 `compute` 准备临时列并为表达式设置最终别名。
- `calculate` 不得使用 `*args`、`**kwargs` 或仅关键字参数。普通业务参数建议由 `compute` 显式传入；纯表达式辅助参数允许引用模块默认常量。
- `compute(df_lazy, ...)` 是唯一外部入口。所有附加参数必须有默认值，且默认值必须引用模块级 `DEFAULT_*` 常量。
- 最终执行链必须精确返回 `trade_time`、`code` 和说明书的 `factor_name`；最终 `.select(...)` 可以位于 `calculate` 或 `compute`，校验器按语义检查而不要求重复选列。
- 自定义参数可能超过模块元数据中的默认 `max_window`。外部调度器必须根据实际参数加载足够的预热数据；元数据 `max_window` 描述默认参数所需窗口。
- 参数只控制公式，不得用于数据加载、日期过滤或触发执行。

### 迁移旧因子的参数收敛

- 旧代码为了试验同时运行多组 `window`、`period` 或 `ewm`，不表示新因子必须暴露全部组合。优先选择一组默认核心参数，并允许调用方覆盖。
- 只有公式逻辑必须同时依赖快慢周期、信号周期或其他联合参数时，才保留多参数组。联合参数使用单个不可变元组传入，避免独立参数错配。
- 区分核心计算窗口与最终结果的外层平滑窗口。核心窗口决定公式，必须保留；单纯为了降噪且未被因子定义明确要求的最终平滑默认移除。
- 若保留外层平滑，它属于因子逻辑的一部分，必须在名称、定义、参数校验和 `max_window` 中体现。

## 迁移去重

- 实现一批旧因子前，先将公式标准化后与目标包及已有因子比较。比较输入字段、变换顺序、窗口统计、分母、符号和最终缩放，不能只比较名称。
- 同一批源代码中完全同式的因子只实现一次；另一个名称记录为重复项，不创建复制文件。
- 与已有因子仅名称不同但完整计算图相同的逻辑不重复实现。若只有默认周期不同，应优先复用或参数化已有逻辑。
- 输入字段相同不代表公式重复；只有完整计算图等价时才能跳过。

指定文件加载器使用 [../assets/file_dataloader.py](../assets/file_dataloader.py)。它对 Feather/Arrow IPC 使用 `pl.scan_ipc`，对 Parquet 使用 `pl.scan_parquet`，并原样保留所有字段名和类型。日期范围、预热数据和多文件编排由外部调用方负责。

## 基差因子输入

基差因子必须使用 [../assets/basis_dataloader.py](../assets/basis_dataloader.py) 准备输入。该加载器同时懒加载期货与现货文件，按 `trade_time + code` 精确连接，并注册以下字段：

- 标识字段：`trade_time`、`code`
- 期货字段：`future_open`、`future_high`、`future_low`、`future_close`、`future_volume`、`future_value`、`future_openint`
- 现货字段：`spot_open`、`spot_high`、`spot_low`、`spot_close`、`spot_volume`、`spot_value`

基差因子只能引用这些注册字段、临时计算别名和最终因子名。最终输出必须为 `['trade_time', 'code', '<factor_name>']`。简单基差通常为 `future_close - spot_close`，基差率通常为 `future_close / spot_close - 1`；分母为零时必须显式处理。

加载器要求期货和现货使用相同的 `code` 表示同一品种，例如都使用 `RB`。连接要求每个 `(trade_time, code)` 最多有一条现货记录；若时间戳不完全一致，不得静默使用精确连接，应另行定义 `join_asof` 方向和容忍时间。

调用示例：

```python
from basis_dataloader import load_basis_data

basis_input_lazy = load_basis_data(
    '/data/future.feather',
    '/data/spot.feather',
)
factor_df = basis_factor.compute(basis_input_lazy).collect()
```

`max_window` 仍保留在元数据中，供外部调度器决定加载长度。它表示计算请求区间首个观测值所需的原始历史观测数量，不只是因子名称中的数字。按最深依赖确定：仅依赖当前周期的计算使用 `1`；N 周期滚动统计通常使用 `N`；使用前收盘价至少需要额外一条历史记录；对 `chgPct` 再做 N 周期滚动通常需要至少 `N + 1` 条原始 `close`；多分支公式取各分支有效依赖的最大值。

## Polars 计算规则

- 默认将每一行同一 `code` 的时序观测称为一个“周期”。当输入可能是分钟、日或其他频率时，元数据、函数文档和注释必须使用“周期”“当前周期”“前一周期”“N周期前”“单周期”等频率中性表述。不得仅根据旧公式或因子名中的 `d`/`w` 推断并写成日频或周频；只有用户明确指定频率时才能使用具体时间单位。
- 默认将 `code` 称为“标的”或“品种”，将 `.over('code')` 描述为“按品种分组”。不得从 OHLCV 字段或旧示例推断资产类别并写成“股票”“个股”“证券”“期货”或“现货”。仅当用户明确指定，或公式使用基差契约中的 `future_*`、`spot_*` 字段时，才使用对应资产类别。
- 执行任何品种序列运算前，使用 `.sort(by=['trade_time', 'code'])`。
- `shift`、`rolling_*`、`cum_*`、`ewm_*`、`diff` 及类似运算必须配合 `.over('code')`。
- 有依赖关系的表达式应拆分到不同的 `.with_columns(...)` 阶段；不要依赖同一阶段刚创建的别名。
- 滚动统计结果再次参与滚动、累计或聚合时，必须先落为临时列，再在下一阶段运算。不得把窗口表达式直接嵌入另一个聚合表达式，否则可能触发 Polars 的 `window expression not allowed in aggregation`。
- 强制使用链式分层：第一阶段产生基础派生列，第二阶段产生窗口或滞后列，第三阶段及以后组合上层结果。层数由依赖图决定，不要求简单公式机械地创建空阶段，但禁止跨层嵌套。
- 每个阶段应使用含义清晰的临时列名显式表达依赖关系，不得用挖掘算子组合隐藏正式因子的阶段拆分。`feature/utils/formula.py` 可以用于探索候选表达式，但候选转为正式因子时必须展开，标准因子文件不能直接导入它。
- 输入频率未明确时，不得直接乘以 `sqrt(252)`、`sqrt(12)` 等固定年化系数。只有用户或数据契约明确给出每年周期数时才允许年化，并应将该约定体现在参数和文档中。
- 全程保持 lazy 转换。因子内部禁止 `.collect()`、`.fetch()`、`.sink_*()` 或其他触发执行和写出的操作；最终执行由外部调用方负责。
- 中间计算使用含义明确的别名，最终表达式使用准确的因子名称。
- 使用 `pl.when(...).then(...).otherwise(...)` 处理零分母或无效分母。只有因子定义允许时才使用 `0.0`，否则使用 `None`。
- 除非因子定义明确要求填充，否则保留预热期自然产生的空值。禁止使用未来信息向后填充。
- 避免 Python 逐行循环、转换为 pandas、全局可变状态和未来数据泄漏。

## 文档与验证

注释使用中文，解释公式、窗口、必需字段、分组方式及不明显的边界处理。窗口注释默认使用频率中性的“周期”，不要把观测窗口固定描述成日或分钟。函数文档必须包含 `df_lazy`、`pl.LazyFrame`、`trade_time`、`code`、公式引用的全部基础字段和准确的输出列。

确保静态校验器和 `py_compile` 均通过。测试必须使用内存 LazyFrame 注入数据，不依赖数据库或日期加载。调用方对返回值执行 `.collect()` 后，检查准确的列顺序、`(trade_time, code)` 唯一性、数值类型、异常无穷值、公式方向与尺度、滞后和窗口边界、品种间隔离，以及不存在未来数据泄漏。

批量因子包还必须检查：每个实际因子文件都在对应批次目录的 `__init__.py` 中使用 `from .xxx import compute as xxx_compute` 显式导出；顶层 `feature/__init__.py` 的 `FactorSpec`、批次索引、分类索引、文件和导出一一对应；不存在遗漏、重复键、错误分类或指向错误实现的别名。
