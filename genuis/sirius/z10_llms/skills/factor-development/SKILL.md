---
name: factor-development
description: 为 Orion 创建、迁移、修改或检查接收外部 LazyFrame 的纯计算 Python 行情因子，并执行分类批次目录、语义因子命名、字段和 Polars 契约。既支持 Codex 直接开发，也支持通过 run_developer.py 调用环境变量配置的大模型生成候选代码；不用于不涉及因子代码的研究或策略分析。
---

# 因子开发

沿用 `demo/xy_factor_history/` 的元数据、命名和 Polars 计算风格，但将数据加载与执行移到因子模块外部。

## 选择运行模式

本 Skill 有两个互斥入口。按用户明确指定的入口执行；用户未指定时，默认使用 Codex 直接开发模式。

### Codex 直接开发模式

- Codex 阅读因子说明书、本 Skill 和相关契约后，直接创建或修改因子代码。
- 使用当前 Codex 的模型能力，不调用 `scripts/run_developer.py`，不读取 `.env`，也不使用 `FACTOR_LLM_*`、`OPENAI_*` 或 `OLLAMA_*` 配置。
- 完成代码后执行本 Skill 规定的静态校验、编译校验和最小运行测试。

### Python 脚本模式

- 仅当用户明确要求运行 `python3 scripts/run_developer.py`、使用外部模型或自动流水线时采用。
- 脚本读取 `factor-specification.json`，默认通过本 Skill 根目录的 `.env` 配置 OpenAI-compatible/Ollama；仅切换配置时传 `--env-file`，进程环境变量优先。
- 脚本把说明书、字段字典和代码骨架序列化为 JSON，再放入标记为不可信、只读的 `<development_input>` XML 数据边界；候选输出仍由确定性代码契约校验。
- 字段字典遵循“显式文件优先、Skill 内置兜底”：传入 `--feature-dictionary` 时，说明书、模型 Prompt、`required_input_fields` 和代码字段引用都只能使用该文件声明的字段；未传入时加载 [references/orion_features.json](references/orion_features.json)。公共键 `trade_time`、`code` 始终允许。
- 候选代码只写入 `--output-dir` 下的隔离目录，不直接发布到正式 `feature/`。运行验证通过可以进入评估；是否进入正式代码目录仍由后续发布授权决定。
- 使用方法、输入输出和模型配置见 [references/pipeline-mode.md](references/pipeline-mode.md)。

## 必须遵循的流程

1. 创建或修改因子代码前，完整阅读 [references/factor-contract.md](references/factor-contract.md)。使用 Python 脚本模式时还必须阅读 [references/pipeline-mode.md](references/pipeline-mode.md)。
2. 检查目标目录和最相近的现有因子。保留用户已有改动，并写入因子家族与市场范围对应的目录；除非用户明确要求，否则不要修改 `demo/xy_factor_history/`。
3. 编码前确认公式、因子家族、市场范围、数据粒度、最终因子名、目标批次目录、所需基础字段及精确回看窗口，并与已有因子做公式级去重；不能只比较名称。默认使用说明书的语义名称；调用方传入 `--factor-name` 时，以该名称覆盖文件名、元数据名、输出列名和下游 artifact 名称，`tf001` 这类合法 snake_case 标识也允许。输入字段只能来自本次实际加载的字段字典；后续新增字段前必须先由用户确认并更新契约。旧公式中的 `preClosePrice`、`chgPct` 和 `vwap` 必须按契约从基础字段派生，当前忽略依赖 `turnoverRate` 的因子。
4. 无参数因子以 [assets/factor_template.py](assets/factor_template.py) 为起点；参数化因子以 [assets/parameterized_factor_template.py](assets/parameterized_factor_template.py) 为起点。替换并移除所有占位符。无参数因子保持简单接口；同一公式仅参数不同的因子必须按契约合并为参数化文件。Python 外部模型模式还会把 [references/generation-patterns.md](references/generation-patterns.md) 作为与成熟 `feature/` 代码一致的生成骨架传给模型。
5. 外部需要从指定 Feather/IPC 或 Parquet 文件供数时，复用 [assets/file_dataloader.py](assets/file_dataloader.py)。加载器必须原样保留字段名和字段类型，不要把加载逻辑复制进因子文件。
   基差因子使用 [assets/basis_dataloader.py](assets/basis_dataloader.py) 同时懒加载期货与现货，并生成已登记的 `future_*`、`spot_*` 输入字段。
6. 使用纯 Polars lazy 表达式实现。处理数学上无效的分母，并确保每个时序运算均按 `code` 排序和隔离。滚动结果再次参与滚动、累计或聚合时，必须拆成多个 `.with_columns(...)` 阶段。因子模块内不得加载或收集数据。
   因子逻辑必须直接使用 `pl.col`、`pl.when`、`with_columns` 等原生 Polars API，保持所有基础特征结构一致。`feature/utils/formula.py` 是供因子挖掘与表达式搜索使用的原生 `pl.Expr` 原子算子库，不是标准因子模块的依赖；正式因子仍须展开为清晰的原生 Polars 链式阶段。
   目标环境不支持已经移除的 `Expr.clip_min()` 和 `Expr.clip_max()`；分别使用 `.clip(lower_bound=...)`、`.clip(upper_bound=...)` 或显式 `pl.when`。
   链式 LazyFrame 是标准实现形式：先用一个 `.with_columns(...)` 生成当前层临时列，再用后续 `.with_columns(...)` 消费这些临时列。任何 `shift`、`diff`、`pct_change`、`rolling_*`、`ewm_*` 或累计结果，只要还要进入另一个窗口、聚合或累计运算，就必须先单独落列；禁止把时序表达式直接嵌套在另一个时序表达式的参数中。
   跨因子包复用的 `safe_div`、`log_return`、参数校验和批量计算必须从 `feature.utils.common` 引用；不得在分类目录中另建 `_common.py` 复制公共逻辑。
   输入频率未明确时，模块元数据、函数文档和代码注释一律使用“周期”描述窗口和滞后，不得默认写成“日”“分钟”“日频”或“分钟频”。资产类别未明确时，统一使用“标的”或“品种”，不得默认写成“股票”“个股”“证券”“期货”或“现货”。只有用户或数据契约明确限定频率或资产类别时，才使用具体表述。
7. 运行 `python3 skills/factor-development/scripts/validate_factor.py path/to/factor.py`。
8. 运行 `python3 -m py_compile path/to/factor.py`。使用内存构造的最小 `pl.LazyFrame` 测试 `compute`，由测试代码执行 `.collect()`，并检查字段结构、唯一性和数值合理性。批量包还必须核对实际因子文件、分类目录 `__init__.py` 显式导出、`FactorSpec` 和顶层 `feature/__init__.py` 注册索引一致且无重复。明确报告未能执行的检查。

## 内部校验与下游门槛

代码验证是本 Skill 的组成部分，不建立独立代码验证 Skill，也不得把未经验证的候选代码交给下游。

验证顺序固定为：生成候选代码、静态契约与编译校验、最小 `LazyFrame` 运行校验。人工代码审核是可选增强，不再是进入运行验证的强制门槛。任何阶段失败都返回本 Skill 内修改；只有运行验证通过并生成 `validated-factor-artifact.json`，且其中 `ready_for_evaluation` 为 `true`，才允许进入 `factor-evaluation`。

最小用法无需审批文件，代码哈希仍由开发 manifest 和运行验证自动核对：

```bash
python3 scripts/finalize_validation.py \
  --run-dir /path/to/development-run \
  --test-data /path/to/test-data.parquet
```

命令成功后生成 `validation_mode: runtime_only` 的下游凭证。若需要人工审批审计，再额外传入 `--approval references/approval.example.json`；审批文件中的 `factor_sha256` 必须与候选文件一致，此时产物记录 `validation_mode: human_approved`。

`scripts/validate_factor.py` 是本 Skill 的内部确定性校验组件，不是独立 Skill。模型不得主观宣布校验通过，未执行的检查必须标记为 `not_run` 或 `pending`。

## 不可变更的输出契约

- 一个文件只实现一种因子逻辑。无参数因子的文件名、元数据名称和输出列名一致；同一逻辑仅窗口或周期参数不同时，使用一个参数化文件一次生成一个或多个带参数后缀的列，不得复制多个实现文件。
- 分类批次目录使用 `<家族><市场范围><三位批次号>`：`t` 表示技术与量价，`m` 表示微观结构；`c` 表示期现通用，`f` 表示期货专属，`s` 表示现货专属，`b` 表示期现联合。例如 `tc001` 是通用技术量价因子的第一批目录，不是因子标识。
- 因子文件名必须是小写 snake_case。文件名、元数据名称、注册名和基础输出列名一致；默认采用说明书语义名称，显式 `--factor-name` 可以使用 `tf001` 这类调用方编号。
- 目录直接使用 `feature/tc001/`、`feature/tf001/`、`feature/ts001/`、`feature/tb001/`；微观结构批次对应使用 `feature/mc001/`、`feature/mf001/`、`feature/ms001/`、`feature/mb001/`。
- 分类依据是公式真实字段依赖和语义。仅使用通用 OHLCV/value 字段的公式属于 `tc`，即使原始代码只在期货上运行；使用 `openint` 或合约专属语义才属于 `tf`；同时使用 `future_*` 与 `spot_*` 才属于 `tb`。
- 模块必须以契约规定的中文元数据文档字符串开头。
- 导入必须包含 `import polars as pl`，不得导入 `dataloader`。
- 每个因子文件使用 `calculate(...)` 实现可独立测试的单次因子公式，使用 `compute(df_lazy, ...)` 作为外部入口。参数化因子的 `calculate` 通常接收一个参数并返回 `pl.Expr`；无参数或必须分阶段计算的复杂因子可以返回 `pl.LazyFrame`。
- 默认参数必须定义为模块顶部的大写不可变常量。`compute` 的参数必须有默认值，并负责参数校验、排序和调用 `calculate`；执行链最终必须精确选列，但该 `.select(...)` 可以在返回 LazyFrame 的 `calculate` 中完成，不要求 `compute` 重复选择。`calculate` 不得遍历参数集合或拼装多个因子列。
- 迁移带有多组试验参数的旧因子时，默认只保留一组能表达核心公式的参数。只有公式本身必须同时依赖多个参数时才保留参数组，并使用有顺序约束的元组。
- 必须区分核心计算窗口和仅作用于最终结果的外层平滑窗口。核心窗口按公式保留；没有明确经济或数学含义的外层平滑默认移除。若平滑属于因子定义，则保留并计入联合参数和 `max_window`。
- `df_lazy` 是外部已经加载、连接并整理好的统一输入；多数据源的连接也必须在调用因子之前完成。
- `compute` 负责调用 `calculate` 并组织最终 `pl.LazyFrame`，不接收日期、不执行 I/O、不调用 `.collect()`。
- 外部调度器负责依据 `数据来源` 和 `max_window` 加载数据、完成多源连接、调用 `compute` 以及最终收集结果。
- 普通行情公共字段为 `trade_time`、`code`、`high`、`low`、`open`、`close`、`volume`、`value`；期货行情另有 `openint`。不得猜测、翻译、重命名或自行创造其他原始输入字段。
- `preClosePrice` 不是基础字段，统一使用 `pl.col('close').shift(1).over('code')` 派生；`chgPct` 不是基础字段，统一使用 `pl.col('close').pct_change().over('code')` 派生。执行前必须先按 `['trade_time', 'code']` 排序。
- `vwap` 不是基础字段，普通行情统一由 `value / volume` 派生；`volume` 为零或空值时结果必须为空值。基差输入分别由对应的 `future_value / future_volume` 和 `spot_value / spot_volume` 派生。
- 当前不登记也不派生 `turnoverRate`；旧因子若依赖该字段，先跳过并明确报告，不得用 `volume` 或 `value` 猜测替代。
- 基差准备数据允许使用公共键 `trade_time`、`code`，以及 `future_open`、`future_high`、`future_low`、`future_close`、`future_volume`、`future_value`、`future_openint`、`spot_open`、`spot_high`、`spot_low`、`spot_close`、`spot_volume`、`spot_value`。这些前缀字段只能由基差加载器从原始字段确定性生成。
- 可以创建公式所需的临时计算列和最终因子列；临时列不得出现在最终结果中。
- 普通因子和基差因子均返回 `trade_time`、`code` 和因子列。
- 所有品种时序计算均按 `['trade_time', 'code']` 排序并使用 `.over('code')`。
- 窗口参数表示观测周期数，与数据频率无关。例如 `rolling_mean(20)` 写作“20周期移动平均”，`shift(1)` 写作“前一周期”，开盘到收盘的同一行变化写作“单周期变化”。
- 数据频率未由用户或契约明确时，不得使用 `sqrt(252)`、`sqrt(12)` 等固定频率年化系数。确需年化时，必须明确每年周期数，不能根据字段或因子名称推断。
- `code` 默认解释为标的或品种标识，`.over('code')` 注释写作“按品种分组”。只有基差等已明确依赖 `future_*`、`spot_*` 字段的场景，才说明期货或现货。
- 中间表达式和最终表达式都使用明确别名，并在最终 `.select(...)` 中移除中间列。
- 不要添加命令行入口、类、框架包装、数据加载、日期过滤、收集、持久化、绘图或策略逻辑。
- 公共辅助函数统一维护在 `feature/utils/common.py`。因子包使用绝对导入 `from feature.utils.common import ...`，避免相对导入到包内重复工具模块。
- 每个批次目录的 `__init__.py` 只使用 `from .mom import compute as mom_compute` 这类语句显式导出，不得建立局部注册表或批量入口。
- `FactorSpec` 和全部注册索引只能定义在最外层 `feature/__init__.py`。同时维护 `BATCH_FACTORS`、`FACTOR_BATCHES` 以及家族、市场范围和数据粒度索引；同一实现可以进入多个索引，但不得复制实现。未指定 `names` 时，顶层批量入口默认只计算 `COMMON_FACTORS`。

`demo/xy_factor_history/` 中内置 `dataloader`、日期参数和 `.collect()` 的部分属于需要纠正的旧结构，不得照搬；其余元数据、命名、字段顺序和 Polars 计算方式仍作为参考。
