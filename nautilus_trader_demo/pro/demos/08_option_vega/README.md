# 认购期权卖方与同月期货对冲策略（08_option_vega）

按固定 Vega 预算卖出虚值认购期权，以实际归属成交仓位计算 Delta，再用同月期货对冲。默认组合为 MO / IM / 000852。回测与 Live 共用 `OptionVegaStrategy` 和 `assembly.py`，只替换输入来源、执行组件和生命周期。

本文按当前入口源码及 `doc/` 专项交接整理，更新日期为 **2026-10-12（Asia/Shanghai）**。操作入口如下：

| 运行方式 | 入口 | 行情与执行 | 结束行为 |
| --- | --- | --- | --- |
| 文件回测 | `run_backtest.py` | 文件分钟行情 + 原生模拟撮合 | 最后实际回测日主动清仓，核对实际仓位和在途订单 |
| Live Recording | `run_live.py --mode recording` | 在线行情 + 目标记录 | 不连接 TD，不产生成交，不保存交易状态文件 |
| Live Model | `run_live.py --mode model` | 在线行情 + 参考价模型成交 | 保存独立模型仓位，停机保留仓位 |
| Live SimNow | `run_live.py --mode simnow` | 在线行情 + 原生 CTP 模拟柜台执行 | 撤销自身活动订单、保存归属仓位，停机保留已成交仓位 |

建议先阅读[回测](#2-文件回测)，再按 [Live 运行方式](#3-live-运行方式)选择执行模式。完整参数分别见回测参数表、[Live 参数表](#4-live-参数说明)；每个代码文件的作用和必要性见[代码职责](#6-代码文件的作用与必要性)。所有命令从 `pro` 根目录执行，使用已安装项目依赖和原生 Bomber 扩展的 Python 环境：

```bash
cd /Users/kerry/work/bomber1/pro
export PYTHONPATH=.
python demos/08_option_vega/run_backtest.py --help
python -m demos.08_option_vega.run_live --help
```

远程运行时将 `cd` 改为部署机器的 `pro` 路径。入口读取 `.env`，环境变量已有值优先；日期、行情表和前置地址必须与实际来源核对。本文提供运行方法，本次文档更新没有连接数据库或柜台。

## 1. 策略逻辑

### 策略思路与候选选择

这是按固定 Vega 预算卖出虚值认购、再用同月期货管理 Delta 的组合策略。它不比较隐含波动率与预测波动率，也不以 IV 高低设置择时门槛；IV 和风险指标主要用于选约、配仓与对冲。

每日默认 13:58，要求认购、同月期货和指数同一分钟价格都已到达。候选须在生命周期内、行权价高于指数、至少剩余 5 个交易日，并能得到有效 Greeks。先保留合格候选中最近的最后交易日，再为各 Delta 档选择最接近的认购；距离相同时按合约代码排序。

同月期货作为 Black76 定价输入，认购真实权利金用于反解 IV。期限按距离最后交易日 15:00 的自然年化时间计算；开平仓天数阈值则使用交易日计数，两者用途不同。

### Vega 预算与卖方手数

设规模基准为 NAV，预算参数为 b，合约乘数为 m。函数中的 Vega 按波动率变化 1 计算：

```text
现金 Vega 预算 B = NAV × b / 100
每一个百分点波动率变化的现金预算 = B / 100
每个 Delta 档的权重 a_i = 1 / (Gamma_i × m_i)
该档未取整手数 = B × a_i / Σ(a_j × Vega_j × m_j)
```

同一认购可匹配多个 Delta 档，先合并数量，再向下取整并应用单合约、总手数上限，最终目标取负数。默认 NAV 40000000、b=1.25，对应每一个百分点的现金预算 5000。NAV 是固定配仓基准，不随回测账户权益更新；每日重新计算替换总目标，不向原仓位累加。取整或手数上限可能使实际 Vega 低于预算。

### 实际持仓对冲与退出

| 环节 | 当前规则 |
| --- | --- |
| 对冲基础 | 使用实际已成交认购仓位，不用尚未成交的目标仓位代替敞口 |
| 分月汇总 | 每个同月期货分别汇总其对应认购的 `实际手数 × 乘数 × Delta` |
| 期货目标 | `round(-汇总敞口 / 期货乘数)`；卖方认购通常对应多头期货保护 |
| 复核触发 | 目标变化、实际认购持仓变化或指定对冲时点 |
| 在途门控 | 相应订单在途时等待；对冲需求超过上限则停止 |
| 认购退出 | 剩余交易日不超过 2，或以指数计算的时间价值不大于零 |
| 风险指标失效 | 仍需持有的认购无法可靠计算 Delta 时停止；待平认购暂保留已有对冲直到实际退出 |
| 最后回测日 | 默认 14:50 起全部期权和期货目标归零，结束核对实际仓位与在途订单 |

成功 IV 可在有效期内缓存，但重算 Greeks 仍使用当前价格与期限；缓存不代替新鲜行情。场次恢复后等待持仓相关腿的新报价，超时会停止，不能将昨日价格当作今天的对冲依据。默认真实价格与 IV 缓存年龄均为 300 秒。

新分钟到达时先处理上一分钟，避免用尚未送达的同帧行情选约。批量组合目标不保证认购与期货同步成交。本例没有额外止损或价格预测，验收重点是卖方方向、Vega 分配、实际成交后对冲、过期报价诊断和末日实际全平。

## 2. 文件回测

### 启动命令

从 pro 根目录、在已安装运行依赖的环境中执行，先设置 `export PYTHONPATH=.`。入口自动读取已有 `.env` 配置；没有配置数据路径时，可通过可选的 `--data-root` 或各文件路径参数指定。示例日期需与实际数据匹配。

**完整执行（显式设置常用参数）：**

```bash
python demos/08_option_vega/run_backtest.py \
  --option-product MO --future-product IM --index-code 000852 \
  --start-day 2026-09-01 --end-day 2026-09-11 \
  --bar-timestamp end --log-level WARNING --no-tearsheet
```

**默认执行（仅必填参数）：**

```bash
python demos/08_option_vega/run_backtest.py \
  --start-day 2026-09-01 --end-day 2026-09-11
```

默认执行采用参数表中的默认值及已有数据路径配置；例如默认启用绩效图，不要求必须产生成交。

### 参数说明

“必选”表示启动时必须传入；“可选”表示有默认值或可由已有配置解析。数据路径虽然是可选参数，但运行前必须能解析到有效文件。

| 参数 | 必选 / 可选 | 含义与默认值 |
| --- | --- | --- |
| `--start-day / --end-day` | 必选 | 必填，请求日期闭区间 |
| `--data-root` | 可选 | 数据根目录，通常包含 role、kline/fut、kline/opt、kline/index |
| `--fut-dir / --opt-dir / --index-dir` | 可选 | 显式覆盖期货、期权、指数真实分钟行情目录 |
| `--fut-basic / --opt-basic` | 可选 | 显式覆盖基础条款文件，默认 role 目录下 fut_basic.feather / opt_basic.feather |
| `--option-product / --future-product / --index-code` | 可选 | 默认 MO / IM / 000852；期权必须有可用同月期货对冲，指数只用于信号 |
| `--bar-timestamp` | 可选 | 可省略，默认 end；源分钟标签为 start 时须显式指定，公共层只移动一次到分钟结束 |
| `--calendar` | 可选 | 可选显式交易日历；未提供时由三类行情全部已知文件日期的并集推导，信任上游交易日完整性，不自动补工作日 |
| `--starting-balance / --nav` | 可选 | 初始资金默认 40000000；固定策略规模基准默认同初始资金，不随账户权益自动更新 |
| `--vega-budget` | 可选 | 默认 1.25；每一个百分点波动率变化的现金风险预算为 nav × 参数 / 10000 |
| `--delta-targets` | 可选 | 默认 0.2,0.25,0.3,0.35，按最接近各档位的 Delta 选择认购；重复合约合并 |
| `--option-slot / --hedge-slots` | 可选 | 每日选约默认 13:58；对冲复核默认 10:00,11:00,13:30,14:00,14:55；实际期权仓位变化也触发对冲 |
| `--min-remaining-days / --close-remaining-days` | 可选 | 开仓至少剩余 5 个交易日，到期前剩余不超过 2 日清仓；计数不含当天、含最后交易日 |
| `--flatten-slot` | 可选 | 最后回测日开始全组合清仓，默认 14:50；指数必须在该时点之后至少有两分钟行情 |
| `--rate / --min-time-value` | 可选 | 默认 0.02 / 5；固定年化利率及允许反解隐含波动率的最少期权时间价值 |
| `--max-market-age-seconds / --max-iv-age-seconds` | 可选 | 默认 300 / 300 秒，真实价格及成功波动率缓存的最长允许年龄 |
| `--max-option-lots / --max-total-option-lots / --max-future-lots` | 可选 | 默认 100 / 200 / 100；单认购、认购总手数、单对冲期货手数上限 |
| `--commission / --max-notional` | 可选 | 每手每次固定手续费默认 1；单订单及单合约持仓名义金额上限默认 100000000 |
| `--margin-init / --margin-maint` | 可选 | 基础期货保证金比例，默认 0.15 / 0.12 |
| `--option-margin-init / --option-margin-maint` | 可选 | 基础期权保证金比例，默认 1 / 1，以权利金名义金额近似 |
| `--require-fills / --no-tearsheet` | 可选 | 可选要求至少一笔成交 / 跳过绩效图 |
| `--index-precision` | 可选 | 指数价格精度默认 4，允许范围 [0,12] |
| `--report-dir` | 可选 | 输出根目录，默认本策略目录的 results/ |
| `--log-level` | 可选 | 框架日志等级，默认 WARNING |

### 数据要求

需要 opt_basic.feather、fut_basic.feather、认购链、同月期货与指数行情。期权最小变动价位直接读取 tickNum，缺失或为空时兼容 minChgPriceNum / price_increment，必须为有限正数；tickUnit 仅为单位说明。未指定 calendar 时用全部已知行情文件日期并集推导交易日；需覆盖期限判断所需的未来日期。末日指数须在清仓时点后至少还有两分钟行情，实际交易腿也须有退出报价和后续撮合机会。

| 数据项 | 要求 |
| --- | --- |
| opt_basic.tickNum | 从每条期权基础条款读取最小变动价位，必须为有限正数；无需命令行或环境变量指定 |

### 输出内容

默认保存到本策略目录的 `results/<run-id>/`，可通过 `--report-dir` 修改根目录。控制台输出数据准备、运行回测、报表与清理的阶段耗时。

| 输出文件 | 内容 |
| --- | --- |
| orders.csv | 模拟委托及订单状态 |
| fills.csv | 实际模拟成交；提交目标不等于成交 |
| positions.csv | 真实合约持仓记录 |
| account.csv | 模拟账户及资金记录 |
| summary.json | 回测摘要、策略参数、信号或调仓统计 |
| input_manifest.json / input_coverage.json / input_issues.json | 输入版本、覆盖与质量诊断 |
| tearsheet.html | 可选绩效图；省略 --no-tearsheet 且绘图依赖可用时生成 |
| signals.csv | 选约、调仓、对冲或退出行为记录 |

### 文件布局、路径优先级与日历

典型文件布局如下，目录日期与文件名日期必须一致，合约代码允许大小写差异：

```text
<data-root>/
  role/fut_basic.feather
  role/opt_basic.feather
  kline/fut/20260901/IM2609_20260901.feather
  kline/opt/20260901/MO2609-C-7300_20260901.feather
  kline/index/20260901/000852_20260901.feather
```

路径优先使用显式 `--fut-dir / --opt-dir / --index-dir / --fut-basic / --opt-basic`。环境变量可以配置 `CTP_DATA_DIR`、`FUT_ROLE_DATA_DIR`、`FUT_KLINE_DATA_DIR`、`OPT_KLINE_DATA_DIR`、`INDEX_KLINE_DATA_DIR`、`KLINE_DATA_DIR` 和 `CTP_CALENDAR_PATH`；专用目录优先于通用根目录推导。`--data-root` 设置根目录，但不会覆盖已配置的资产专用环境变量。完整兼容规则见[公共输入层](../../bomber/framework/dataprep/README.md#路径配置)。

分钟文件需要时间列（`datetime`，兼容 `timestamp`）和 `open/high/low/close/volume`，`volume` 为分钟增量；按日盘规则校验时间、身份和 OHLCV，不前填价格或补造分钟。`--bar-timestamp start` 会在准备层加一分钟，默认 `end` 不移动。该参数同时用于三类文件，因此上游标签需一致。期权基础条款需要合约代码、类型、行权价、乘数、标的指数、交易所和上市/最后交易/到期日期；字段契约见[基础条款说明](../../doc/data/BASIC_METADATA_ALIGNMENT.md)。

可从 Live 使用的公共日历导出同版本文件，避免由缺文件误判休市：

```bash
python -m scripts.integration.reference_data.calendar_export \
  --calendar-id china.cffex --start 2026-01-01 --end 2026-12-31 \
  --output demos/08_option_vega/results/cffex-2026.csv

python demos/08_option_vega/run_backtest.py \
  --data-root /path/to/market-data \
  --calendar demos/08_option_vega/results/cffex-2026.csv \
  --start-day 2026-09-01 --end-day 2026-09-11 \
  --bar-timestamp end --require-fills --no-tearsheet
```

`/path/to/market-data` 替换为实际根目录。导出 CSV 旁的 manifest 保存规则版本与摘要，回测加载时核验；普通 CSV/Feather 日历仍兼容。不传日历时由三类行情**全部已知文件日期的并集**推导，默认信任上游完整性；应让日期覆盖结束日之后足够多的交易日，以判断剩余期限。

### 回测指数如何接入、未传路径为何仍能运行

**当前 08 回测读取本地指数 Feather 分钟文件。** Live 的 `dolphindb-dfs` 指数使用声明式特征 Provider、Schema 与查询缓存；回测入口目前没有调用这条特征查询链，也不使用 Live 的 `features.json` 或 `--index-source` 参数。两个入口最终都向同一个策略提供指数价格，但输入装配不同：

| 场景 | 指数输入链路 | 策略如何接收 |
| --- | --- | --- |
| 当前文件回测 | 本地 `000852_YYYYMMDD.feather` → 公共 dataprep 校验/标准化 → `FileReplayFeed` | `DataBinding("000852", …, DataType.BAR, "1-MINUTE")` → `on_bar` → `latest["000852"]` |
| Live DFS 指数 | DolphinDB 原表 → 声明式 `index.minute` 特征 Provider/缓存 → `FeatureQueryBarFeed` → 封闭分钟帧 | `on_market_frame` → `latest["000852"]` |
| Live 标准指数流 | DolphinDB 已收盘 Bar 流 → 独立 Feed → 封闭分钟帧 | `on_market_frame` → `latest["000852"]` |

指数只用于选约、时间价值及清仓时钟，没有执行路由，不作为可成交期货加入账户。当前回测中，指数、期权和期货共用名为 `option-bars` 的回放 Feed 绑定；这个名字不表示其中只有期权。

回测代码位置按调用顺序如下：

| 位置 | 做什么 |
| --- | --- |
| [run_backtest.py](run_backtest.py)：模块初始化的 `load_dotenv()`、`run_case` 中 `resolve_option_args` | 加载环境配置，解析并验证 `args.index_dir` |
| [dataprep/paths.py](../../bomber/framework/dataprep/paths.py)：`resolve_paths / resolve_option_args` | 指数优先级：显式 `--index-dir` → `INDEX_KLINE_DATA_DIR` → `KLINE_DATA_DIR/index` → 根目录的 `kline/index`；根目录可来自 `--data-root` 或 `CTP_DATA_DIR`；无这些来源时兼容 `KLINE_DIR/index` 或旧直接目录 |
| [run_backtest.py](run_backtest.py)：`idx_paths = inventory(...)`、每日 `needed[(day, args.index_code)]` | 索引目标指数文件，缺少任一回测日指数文件会抛异常 |
| [run_backtest.py](run_backtest.py)：`prepare_option_chain(...)`；[scenarios/option_chain.py](../../bomber/framework/dataprep/scenarios/option_chain.py)：`prepare_option_chain` | 把指数作为 `asset_kind="index"` 加入输入计划，读取文件并校验时间和 OHLCV |
| [run_backtest.py](run_backtest.py)：`index_id` 注册及 `add_prepared_bar_source`；[dataprep/bars.py](../../bomber/framework/dataprep/bars.py)：`add_prepared_bar_source` | 注册指数行情元信息；将已准备数据用 `PreparedFrameReader + FixedInstrumentBarParser` 加入回放，不再读取原文件或重复移动标签 |
| [assembly.py](assembly.py)：`all_ids`、`DataBinding`；[strategy.py](strategy.py)：`on_bar` | 绑定包括指数在内的所有行情，回放送入策略缓存并参与分钟决策 |

因此，不显式传 `--index-dir` 也能运行，表示已有配置解析出了有效目录、目录中已有指数文件；不能据此推定没有指数输入。该入口不会自动下载指数、查询 DolphinDB 补数据、生成假指数或用 IM 价格替代指数。

2026-10-12 用户返回的 9 月 1—9 日回测成功日志已通过文件校验、每日选约三腿同步、最终结果检查，说明该运行环境具备所需指数输入。**控制台未展示实际目录，当前尚不能仅凭日志确定是哪个环境变量提供了它。** 已保存的输入报告可直接回答这个问题，比读取当前可能已经变化的 `.env` 更准确。

在运行机 `pro` 目录执行以下只读命令，读取本次成功报告中的路径和来源，不输出账户凭据：

```bash
python - <<'PYCODE'
import json
from pathlib import Path
report = Path("demos/08_option_vega/results/70d4ca7d-e457-4f78-a7f2-0b2a4f7d7bed")
manifest = json.loads((report / "input_manifest.json").read_text())
coverage = json.loads((report / "input_coverage.json").read_text())
for entry in manifest.get("paths", []):
    print("指数目录:", entry.get("resolved", {}).get("index"))
    print("路径来源:", entry.get("sources", {}).get("index"))
for filename, info in coverage.get("files", {}).items():
    if "asset_kind='index'" in info.get("identity", ""):
        print("指数文件:", filename, "行数:", info.get("rows"))
PYCODE
```

来源为 `env:INDEX_KLINE_DATA_DIR` 表示专用环境变量；`derived` 表示由行情父目录或根目录推导；`legacy_kline` 表示旧目录兼容解析。这些环境值可能由已有进程环境或 `.env` 提供，manifest 的 `env:` 本身不区分两者。`input_coverage.json` 另记录每个文件的行数、首末纳秒时间与标签，`input_manifest.json` 记录实际读取文件的大小与修改时间。

### 回测完成条件与模拟边界

每天选约分钟至少要有一组认购、对应期货、指数同步行情；缺日数据会报错，不自动截短请求区间。最后实际回测日的指数最大时间须不早于 `flatten-slot + 2分钟`，持仓腿还需有退出报价及后续撮合事件。报告中 `remaining_positions_or_working` 应为空，`report_errors` 应为空；未触发最终清仓或有残留则失败。`--require-fills` 额外要求至少一笔模拟成交。

模拟使用 NETTING、Bar 撮合、每手固定手续费和权利金比例近似期权保证金；订单最早在后续行情事件撮合，组合多腿不保证原子成交。没有完整期权卖方保证金、行权和到期结算模型；实际成交、资金和风险应结合这些范围解释。

### 排查“持仓计算缺少新鲜行情”

2026-10-12 返回的 9 月 1—11 日回测日志在 `2026-09-10 09:37` 停止：IM2609 实际多头 2 手、目标 2 手、在途 0，策略最新报价为 `2026-09-09 15:01`，年龄 66960 秒。隔夜恢复已等待超过默认 300 秒，不能继续按昨日价格计算持仓。日志里的 IM2611 未上市、IM2706 缺基础条款是候选排除记录，当前异常指出的合约是 IM2609。

文件加载和每日 13:58 同步检查通过，只证明当天文件及选约时点有覆盖，不证明开盘已有持仓腿的新报价。先在同一个运行环境、相同 `.env` 和路径参数下执行只读诊断，无需再次启动引擎：

```bash
python demos/08_option_vega/check_backtest_data.py \
  --day 2026-09-10 --symbol IM2609 --at 09:37 --bar-timestamp end
```

若原回测提供了 `--data-root / --fut-dir / --index-dir`，检查命令也传相同值。它复用公共路径解析、文件索引、分钟标签转换及 OHLCV 校验，输出实际路径、原始/完成分钟范围、决策前最新价格时间、后续首条和附近样例；零成交量行保留。检查范围为原文件覆盖，不能独立证明回放送达。

| 诊断结果 | 处理方向 |
| --- | --- |
| `FILE_MISSING / DIRECTORY_MISSING` | 核对实际解析目录、文件名与日期；补齐持仓合约的真实输入 |
| `INPUT_ERROR` | 按错误核对时间列、日期身份、重复分钟及 OHLCV |
| `NO_BAR_BY_DECISION` | 当天文件存在但决策前没有完成分钟，检查采集/导出覆盖和原始标签 |
| `STALE_AT_DECISION` | 当天最近分钟已超过允许年龄，核对缺口；恢复还要求首条行情属于当前场次 |
| `FRESH_AT_DECISION` | 文件中有新鲜行；进一步对照原失败报告的路径/文件版本和回放绑定、送达记录 |

诊断返回非零表示所查分钟存在缺失、过期或输入错误，JSON 仍会输出。源分钟标签必须按数据实际含义设置；`start/end` 的一分钟移动不能填补缺失的开盘行情。提高年龄上限不能恢复缺失数据，且可能使昨日价格继续参与风险计算。保留失败目录里的 `input_manifest.json / input_coverage.json / input_issues.json`，用于核对实际文件来源与版本；该 9 月 1—11 日区间仍未通过；后续 9 月 1—9 日成功结果见第 7 节，缩短区间没有修复 9 月 10 日缺报价问题。

## 3. Live 运行方式

### 3.1 准备来源与环境

行情、指数、参考资料和执行模式分别配置，来源不会隐式回退：

| 输入 | 可选来源 | 运行要求 |
| --- | --- | --- |
| 期权及同月期货 | `--market-source ctp`（默认） | CTP 原始深度聚合一分钟 Bar；`last` 使用成交，`mid` 使用盘口中间价 |
| 期权及同月期货 | `--market-source dolphindb` | 显式提供已收盘期权/期货标准分钟流表 |
| 指数 | `--index-source dolphindb`（默认） | 独立标准分钟流表，只有行情绑定，无交易路由 |
| 指数 | `--index-source dolphindb-dfs` | 只读查询配置声明的指数分钟原表，默认 `dfs://min_bar/index_1min` |
| 参考资料 | `--reference-source dolphindb`（默认） | 读取期权、期货条款快照；按估值日检查有效性 |
| 参考资料 | `--reference-source file` | 仅显式测试：同时提供 `--allow-file-reference-test --opt-basic ... --fut-basic ...` |
| 日历 | `--calendar-id china.cffex`（默认） | 公共编译日历；Live 不接受 `--calendar` 文件参数 |

项目 `.env` 或运行环境需要以下配置；不要把密码放进命令或报告：

| 环境变量 | 用途与默认值 |
| --- | --- |
| `DDB_HOST / DDB_USERNAME / DDB_PASSWORD` | DolphinDB 连接必需；指数及默认参考资料共用这些凭据 |
| `DDB_PORT / DDB_STREAMING_PORT` | 连接端口默认 8848；流订阅监听端口默认 0 |
| `DDB_REFERENCE_DATABASE` | 参考资料库默认 `dfs://bomber_daily`；可用 `--reference-database` 覆盖 |
| `DDB_FUT_BASIC_TABLE / DDB_OPT_BASIC_TABLE` | 参考表默认 `fut_basic / opt_basic` |
| `DDB_INDEX_BAR_TABLE` | 标准指数流表，或用 `--index-bar-table`；DFS 模式不使用此变量 |
| `DDB_OPTION_BAR_TABLE` | DolphinDB 期权/期货流表，或用 `--market-bar-table` |
| `CTP_MD_ADDRESS` | `market-source=ctp` 必需，按 realtime / replay 选择实际核验的行情前置 |
| `CTP_ACCOUNT_ID / CTP_PASSWORD` | CTP MD 或原生 TD 登录必需 |
| `CTP_BROKER_ID` | MD 默认 9999；SimNow TD 必须显式配置为 9999 |
| `CTP_TD_ADDRESS` | 仅原生 SimNow 执行必需 |
| `CTP_APP_ID / CTP_AUTH_CODE` | TD 认证配置，按实际账户填写 |
| `CTP_PRODUCTION_MODE` | 默认 true，第二套看穿式前置要求 true；这是连接配置，不代表真实资金生产授权 |
| `CTP_MD_FLOW_PATH / CTP_TD_FLOW_PATH` | 可选覆盖公共客户端 flow 目录；默认使用带进程 ID 的 `/tmp` 目录 |

DolphinDB 标准流遵循 `DolphinDbStreamSpec.bar()` 的字段和顺序：

```text
minTime,date,Code,open,high,low,close,prev_close,prev_settlement,
volume,turnover,pct_change,open_interest,prev_open_interest,updateTime
```

DFS 指数原表由[特征配置](../../bomber/framework/dataprep/features/configs/features.json)声明，字段为 `minTime/date/Code/open/high/low/close/prev_close/volume/turnover/pct_change/updateTime`，无需冒充标准流表。查询验证真实类型，按上海时区转换 UTC 纳秒；价格/数量无法按配置精度无损表达时失败。公共查询、只读探针及导出方法见[特征说明](../../bomber/framework/dataprep/features/README.md)。

### 3.2 实时 Recording：直接运行

以下采用 CTP 成交分钟 + DFS 指数；先将 `VEGA_DAY` 设置为已核验的当天 MD 交易日，并确保 `.env` 中 MD 地址为实时前置。示例日期为 20261012，运行时替换。所有时点均为上海时间，需在选约分钟前预留连接、首根完成 Bar 和就绪时间。

```bash
export VEGA_DAY=20261012

# 同一终端保存公共参数；后续 Model / SimNow 命令复用。
VEGA_LIVE_ARGS=(
  --connect --market-source ctp --ctp-price-type last
  --simnow-environment realtime --expected-trading-day "$VEGA_DAY"
  --valuation-day "$VEGA_DAY" --calendar-id china.cffex
  --reference-source dolphindb
  --option-product MO --future-product IM --index-code 000852
  --index-source dolphindb-dfs
  --index-database dfs://min_bar --index-table index_1min
  --index-bar-timestamp end --index-availability observed
  --index-price-precision 4 --index-size-precision 4
  --index-poll-seconds 0.2 --index-query-timeout 2
  --option-slot 13:58 --delta-targets 0.25
  --frame-wait-seconds 5 --instrument-max-age-seconds 30
  --ready-timeout 180 --seconds 900
  --max-option-lots 1 --max-total-option-lots 2 --max-future-lots 1
  --max-notional 2000000
)

python -u -m demos.08_option_vega.run_live \
  "${VEGA_LIVE_ARGS[@]}" --mode recording \
  --report-dir demos/08_option_vega/results/live-recording
```

数组语法适用于 Bash/Zsh。在 13:53 左右启动该 900 秒示例可覆盖默认 13:58 选约；若提前很久启动，应增大 `--seconds`。小手数示例显式用单档 `--delta-targets 0.25`：默认四档可能选到多个不同合约，总手数上限 2 导致比例缩小后全部向下取整为零。单档有利于验收目标链，但仍需满足行情和选约条件；不改变策略默认四档。

Recording 不接受 `--state-file / --resume / --require-fills / --enable-orders / --confirm-simnow`。它没有成交，无法证明实际成交后 Delta 对冲。

### 3.3 实时 Recording：自动运行与核验

自动入口固定 MO / IM / 000852、CTP LAST、DFS 指数、实时环境、默认四档 Delta，继承 `.env` 的实际前置配置；它不复用上面的数组，也不连接 TD。默认应在 13:53 前启动：

```bash
python -u -m scripts.integration.options.recording \
  --trading-day "$VEGA_DAY" --seconds 4200 \
  --report-dir demos/08_option_vega/results/recording-acceptance
```

必须在目标交易日、选约时点前至少 `ready-timeout + 65秒` 启动，时长要覆盖选约分钟及等待时间。自动入口另支持 `--option-slot`（默认 13:58）、`--ready-timeout`（默认 180 秒）、`--summary` 和 `--allow-no-target`。自定义选约时点须早于策略默认清仓时点 14:50。

它从唯一子进程 `summary.json` 检查实际 MD 日、编译日历、指数查询、同分钟接收期限和实际非零 Recording 请求，写出 `recording-acceptance.json`。运行报告 `status=passed` 只说明运行器验收通过，不自动代表非零目标链通过；默认四档取整为零时目标链核验会失败。

仅复核已有报告，无网络连接：

```bash
python -m scripts.integration.options.recording \
  --summary /path/to/summary.json --trading-day "$VEGA_DAY"

# 缺少非零目标时，只检查决策评价；验收范围为 decision_only。
python -m scripts.integration.options.recording \
  --summary /path/to/summary.json --trading-day "$VEGA_DAY" --allow-no-target
```

自定义选约时点的报告需传相同 `--option-slot`。自动核验排除 Replay 工程映射，不用工程结果登记实时通过。

### 3.4 Model：在线行情驱动模型成交

在同一终端复用 3.2 的参数，每次新会话使用尚不存在的独立状态文件：

```bash
VEGA_MODEL_STATE="demos/08_option_vega/results/model-vega-$(date +%Y%m%d-%H%M%S).json"
python -u -m demos.08_option_vega.run_live \
  "${VEGA_LIVE_ARGS[@]}" --mode model \
  --strategy-position-source model --account-reconciliation off \
  --state-file "$VEGA_MODEL_STATE" --require-fills \
  --report-dir demos/08_option_vega/results/live-model
```

Model 按公共执行参考价模拟成交，维护独立策略账本，不查询物理账户，也不接受报单授权开关。`--require-fills` 要求本次出现策略归属成交；未覆盖选约时点、配仓为零或缺少行情会导致成交验收失败。模型成交不证明盘口可成交性。

### 3.5 SimNow：原生模拟柜台执行

配置实际 TD 前置及账户，并核验 TD 交易日。实时示例中 `VEGA_DAY` 须与 TD / MD 日一致；默认 strict 新会话要求本次管理合约范围内没有待接管仓位。已经存在的本策略仓位使用原文件恢复，不能用新文件绕过归属检查。

```bash
VEGA_CTP_STATE="demos/08_option_vega/results/ctp-vega-$(date +%Y%m%d-%H%M%S).json"
python -u -m demos.08_option_vega.run_live \
  "${VEGA_LIVE_ARGS[@]}" --mode simnow \
  --enable-orders --confirm-simnow \
  --strategy-position-source fills --account-reconciliation strict \
  --state-file "$VEGA_CTP_STATE" --require-fills \
  --limit-offset-ticks 1 --max-session-orders 4 \
  --report-dir demos/08_option_vega/results/live-simnow
```

该命令会向 SimNow 模拟柜台报单。`--enable-orders` 和 `--confirm-simnow` 缺一不可；仅授权此受控 SimNow 入口。示例把单会话报单上限设为 4，入口默认 20。到达上限会进入停机撤单和保存流程，不自动平仓。

限价以完成分钟执行参考价及 `--limit-offset-ticks` 计算，不以实时买一/卖一重新定价。`order_submission_audit` 和 INFO“CTP报单诊断”记录限价、分钟参考价、已观测盘口、年龄和是否跨价；盘口过期或缺失时跨价为空。提交调用返回不代表柜台接受，柜台接受或观测跨价也不保证成交；成交以实际归属回报为准。CFFEX 使用通用 CLOSE。

### 3.6 替换为 DolphinDB 标准分钟流

在 3.2 数组中替换对应来源参数，并用实际已核验表名：

```text
--market-source dolphindb --market-bar-table <期权与期货已收盘分钟流表>
--market-bar-timestamp end
--index-source dolphindb --index-bar-table <指数已收盘分钟流表>
--index-bar-timestamp end
```

两类来源可以独立选择，例如 CTP 市场 + DolphinDB 指数流。标准流没有默认表名，必须显式参数或环境变量配置，且遵循标准 Bar 字段契约。文件不作为 Live 当前指数输入，期货价格不能代替指数。全 DolphinDB 行情无需 MD 连接；原生执行仍需 TD 配置，使用实时环境。

### 3.7 Replay 与工程时间映射

**原始时间 Replay**：使用 `--simnow-environment replay --replay-md-trading-day YYYYMMDD`。估值日默认为原始 MD 日，选约和 Black76 按原始事件时间；执行请求使用接收墙钟。只豁免历史事件相对墙钟的年龄，仍检查 MD 日、源日期、接收健康、倒退和参考条款。期权、期货、指数必须属于同一估值日；墙钟到 13:58 不会触发源流其他时刻的选约。

**工程映射**：`--replay-test-start-time 13:55` 将首份完成 CTP 分钟 Bar 映射到估值日 13:55，所有合约共享固定偏移，历史指数按映射后的同分钟查询。可另设 `--replay-test-source-time 16:00`，把源 16:00 对应到目标开始时刻。保留分钟间隔、OHLCV 和接收时间，不补造成交/指数，源日期或映射估值日跨日会失败。

以下采用历史证据中的 MD / TD 日 20261009，仅作配置示例；运行前应重新核验当前环境返回的日期、前置及历史指数覆盖：

```bash
export VEGA_MD_DAY=20261009
export VEGA_TD_DAY=20261009
# CTP_MD_ADDRESS 设为已核验 Replay 前置，CTP_PRODUCTION_MODE=true。
python -u -m demos.08_option_vega.run_live \
  --connect --mode recording --market-source ctp --ctp-price-type last \
  --simnow-environment replay --replay-md-trading-day "$VEGA_MD_DAY" \
  --expected-trading-day "$VEGA_TD_DAY" --valuation-day "$VEGA_MD_DAY" \
  --replay-test-start-time 13:55 --option-slot 13:58 --delta-targets 0.25 \
  --calendar-id china.cffex --reference-source dolphindb \
  --option-product MO --future-product IM --index-code 000852 \
  --index-source dolphindb-dfs --index-database dfs://min_bar --index-table index_1min \
  --index-bar-timestamp end --index-availability observed \
  --index-poll-seconds 0.2 --index-query-timeout 2 \
  --index-price-precision 4 --index-size-precision 4 \
  --frame-wait-seconds 5 --instrument-max-age-seconds 30 \
  --ready-timeout 180 --seconds 600 \
  --max-option-lots 1 --max-total-option-lots 2 --max-future-lots 1 \
  --max-notional 2000000 \
  --report-dir demos/08_option_vega/results/replay-recording
```

删除 `--replay-test-start-time` 即恢复原始时间路径，运行仍须覆盖源流选约时点。工程映射仅支持 **CTP Replay + DFS 指数 + 新会话**，不能 `--resume`。Model 可替换 mode 并按 3.4 提供新状态文件、`--require-fills`；SimNow 可按 3.5 配置报单授权、独立新状态和实际 TD 日。原生还要求所选历史合约在实际 TD 日仍可交易。

报告标记 `validation_scope=engineering_chain`、`bar_time_basis=replay_test_mapped`，保存映射偏移和原始/映射时间样例。映射价格流与历史指数不是同步原市场行情，只验证接收、聚合、分钟帧、信号、执行与归属链路，不能用来评估历史策略表现或登记实时验收。

## 4. Live 参数说明

`run_live.py` 的默认值与回测独立。除 `--connect`（运行必须显式提供）和 `--expected-trading-day` 外，其他参数按所选模式有条件必需；完整关系见下表。日期使用 `YYYYMMDD`，日内时点使用 `HH:MM`。

### 4.1 策略参数与回测差异

| 参数 | Live 默认值 | 用途 / 与回测差异 |
| --- | --- | --- |
| `--option-product / --future-product / --index-code` | MO / IM / 000852 | 期权/同月期货/独立指数；Live 指数必须为六位数字 |
| `--nav` | 40000000 | 固定配仓规模；Live 无 `--starting-balance`，不代表柜台余额 |
| `--vega-budget` | 1.25 | 每一百分点波动率变化的现金预算为 `nav × b / 10000` |
| `--delta-targets` | 0.2,0.25,0.3,0.35 | 逗号分隔，每档在 (0,1)；重复合约合并，小手数验收可显式用 0.25 |
| `--option-slot` | 13:58 | 每日选约时点，必须早于 `flatten-slot` |
| `--hedge-slots` | 10:00,11:00,13:30,14:00,14:55 | 逗号分隔的对冲复核时点；实际归属持仓变化也触发 |
| `--flatten-slot` | 14:50 | Live 仅在配置清仓日时执行最终组合清仓；回测默认末日清仓 |
| `--flatten-day` | 不设置 | 主动清仓日，须为估值日当天或之后的已知交易日；不设置则停机保留仓位 |
| `--min-remaining-days / --close-remaining-days` | 5 / 2 | 开仓最少剩余交易日 / 清仓阈值；须满足 `0 ≤ close < min` |
| `--rate / --min-time-value` | 0.02 / 5 | Black76 年化利率 / 反解 IV 的最少时间价值，后者非负 |
| `--max-market-age-seconds / --max-iv-age-seconds` | 300 / 300 | 策略真实价格 / 成功 IV 缓存最长年龄，正整数秒 |
| `--max-option-lots` | **1** | 单认购上限，回测默认 **100** |
| `--max-total-option-lots` | **2** | 认购总手数上限，回测默认 **200** |
| `--max-future-lots` | **1** | 单对冲期货上限，回测默认 **100**；超限停止 |
| `--max-notional` | **2000000** | 单订单及单合约持仓名义金额上限，回测默认 **100000000** |
| `--require-fills` | false | Model/SimNow 要求本次归属成交；Recording 禁止；回测要求模拟成交 |
| `--report-dir` | 本策略 `results/` | 报告根目录；状态文件另由 `state-file` 指定 |
| `--log-level` | INFO | DEBUG/INFO/WARNING/ERROR；回测默认 WARNING |

Live 不接受回测的 `--start-day / --end-day / --data-root / --fut-dir / --opt-dir / --index-dir / --calendar / --bar-timestamp / --index-precision / --commission / --margin-init / --margin-maint / --option-margin-init / --option-margin-maint / --no-tearsheet`。Live 的 `--opt-basic / --fut-basic` 仅用于显式文件参考测试；指数使用下面的独立精度及时间标签参数。

### 4.2 模式、交易日、执行和恢复

| 参数 | 默认值 | 含义与约束 |
| --- | --- | --- |
| `--connect` | false | 所有 Live 运行必须显式提供，包括 Recording/Model |
| `--mode` | recording | recording / model / simnow，分别对应记录、模型、原生柜台 |
| `--expected-trading-day` | 必填 | 执行/参考会话交易日；SimNow 与实际 TD 日核对 |
| `--valuation-day` | 自动 | realtime 默认会话日，replay 默认 MD 日；必须与所选行情环境日期一致 |
| `--simnow-environment` | realtime | realtime / replay；用于 CTP 行情时间政策，与执行 mode 独立 |
| `--replay-md-trading-day` | 无 | replay 必需的原始 MD 日，realtime 禁止 |
| `--replay-test-start-time` | 无 | 工程映射目标开始分钟；仅 CTP Replay + DFS 指数新会话 |
| `--replay-test-source-time` | 无 | 显式源分钟，必须同时设置 start-time；省略则以首份完成 Bar 为锚 |
| `--calendar-id` | china.cffex | 编译日历身份、覆盖和指纹写入报告/恢复身份；现维护到 2026 年，超范围拒绝 |
| `--seconds` | 900 | 就绪后运行时长，有限正数秒；应覆盖源流选约时点 |
| `--ready-timeout` | 90 | 启动等待就绪期限，有限正数秒 |
| `--query-timeout` | 30 | 原生 TD 查询/连接超时，有限正数秒 |
| `--limit-offset-ticks` | 1 | 原生限价相对执行参考价的偏移跳数，非负整数 |
| `--max-session-orders` | 20 | 原生单会话报单上限，正整数；达到后停机撤单保存 |
| `--account-reconciliation` | 按模式 | Recording 强制 off；Model 只允许 off；SimNow 默认 strict，可选 observe/off |
| `--strategy-position-source` | 按模式 | Model 默认且必须 model；Recording/SimNow 默认且必须 fills；不把物理总仓当策略仓位 |
| `--enable-orders / --confirm-simnow` | false / false | SimNow 两个都必须显式提供；Recording/Model 禁止 |
| `--state-file` | 无 | Model/SimNow 必需：新会话文件须尚不存在；Recording 禁止 |
| `--resume` | false | 沿用已存在的同一原状态文件；工程映射/Recording 禁止 |

SimNow 的 strict 将柜台仓位与执行账本核对，不一致闭闸；observe 只观测差异；off 不查询柜台仓位。各模式的策略仍只读取自身归属成交和自身未成交委托分配；改变政策不自动接管柜台已有仓位。

### 4.3 行情、参考资料与分钟同步

| 参数 | 默认值 | 含义与约束 |
| --- | --- | --- |
| `--market-source` | ctp | ctp / dolphindb，期权及同月期货来源 |
| `--index-source` | dolphindb | dolphindb 标准流 / dolphindb-dfs 原表查询 |
| `--ctp-price-type` | last | last 成交 Bar；mid 报价 Bar，volume=0；无成交 LAST 不造 Bar |
| `--market-bar-table` | 无 | DolphinDB 市场标准流表，或 `DDB_OPTION_BAR_TABLE`；该来源必需 |
| `--index-bar-table` | 无 | DolphinDB 指数标准流表，或 `DDB_INDEX_BAR_TABLE`；标准流来源必需 |
| `--market-bar-timestamp` | end | DolphinDB 市场流 start/end 标签；CTP 聚合时间由适配器规范化 |
| `--index-bar-timestamp` | 不覆盖 | start/end；DFS 沿配置，内置 end；标准流默认 end |
| `--frame-wait-seconds` | 2 | 同一分钟收集期限，有限秒数，`0 ≤ 值 ≤ min(30,max-market-age-seconds)` |
| `--instrument-max-age-seconds` | 30 | CTP 必需合约深度接收时效，`0 < 值 ≤ 120` 秒，独立于策略 Bar 年龄 |
| `--reference-source` | dolphindb | dolphindb / file；来源显式绑定 |
| `--reference-database` | 环境/默认 | 覆盖 `DDB_REFERENCE_DATABASE`，默认 `dfs://bomber_daily` |
| `--reference-timeout` | 30 | 参考资料读超时，正整数秒 |
| `--reference-refresh-seconds` | 5 | 控制线程刷新间隔，有限非负数秒 |
| `--reference-max-observation-seconds` | 30 | 资料最大观测年龄，有限正数秒 |
| `--allow-file-reference-test` | false | 文件参考测试必须显式提供 |
| `--opt-basic / --fut-basic` | 无 | file 参考模式两个文件都必需；仅条款，行情仍在线 |

行情帧的 `valuation_ns` 决定选约日期/时点和 Greeks；`request_ns` 使用接收墙钟用于执行时效。无行情候选不参与选约，持仓腿/对应期货/指数失效则等待恢复或失败闭闸。封闭后迟到 Bar 不追溯修改决策。DFS 指数要求同分钟，期限内缺失会跳过**整次决策**并增加 `skipped_index_frames`；晚到不补做。

### 4.4 DFS 指数查询

以下参数用于 `index-source=dolphindb-dfs`：

| 参数 | 默认值 | 含义与约束 |
| --- | --- | --- |
| `--index-feature-config` | `bomber/framework/dataprep/features/configs/features.json` | 数据集、原表字段、时间与 Bar 投影配置 |
| `--index-dataset / --index-schema-version` | index.minute / 1 | 数据集身份及正整数 Schema 版本；必须有一分钟真实 OHLCV 投影 |
| `--index-database / --index-table` | 沿配置 | 内置 `dfs://min_bar / index_1min`，可显式覆盖 |
| `--index-availability` | 沿配置 | 内置 observed；source-time 使用源发布时间，update-time 是 source-time 的兼容值 |
| `--index-poll-seconds` | 0.2 | 查询刷新间隔，`0 < 值 ≤ 30` 秒；不保证数据库实际延迟 |
| `--index-query-timeout` | 2 | 查询超时，整数 [1,30] 秒 |
| `--index-price-precision` | 4 | 指数价格精度 [0,9]，也用于标准指数流元信息 |
| `--index-size-precision` | 4 | DFS 数量精度 [0,9]；标准指数流数量精度固定 0 |

查询在控制线程、分钟帧锁外执行，只将市场待决策分钟的同分钟指数 Bar 投入缓冲；启动不灌入历史整日 Bar。没有估值日数据时等待至 ready-timeout，连接、类型、内容冲突失败闭闸。

`observed` 以本次查询完成作为首次可见时间，重复读取保持首次观测，内容改变形成新版本；不能恢复下载前的历史可用性。只有来源确认 `updateTime` 是真实发布/修订可用时间时才选 `source-time`。原表仅保留最后覆盖值时，无法重建消失的历史修订；历史研究应使用带版本与可用时间的标准导出。

## 5. 恢复、停机与输出

### 恢复与主动清仓

Model/SimNow 使用原始时间路径、原执行模式和同一原文件恢复，在原完整启动命令中增加：

```text
--resume --state-file <上一次的原状态文件>
```

例如恢复 3.4 的实时 Model 会话，保持原参数数组并明确指定原文件：

```bash
VEGA_RESUME_STATE=/path/to/original-model-state.json
python -u -m demos.08_option_vega.run_live \
  "${VEGA_LIVE_ARGS[@]}" --mode model \
  --strategy-position-source model --account-reconciliation off \
  --resume --state-file "$VEGA_RESUME_STATE" \
  --report-dir demos/08_option_vega/results/live-model-resume
```

恢复阶段可继续提供 `--require-fills`，但该参数要求**恢复后本次新增成交**，原文件中的已有仓位不计作本次成交。SimNow 恢复还需保留原生报单授权、原账户政策和其他原始参数。

保留已成交归属仓位；首次新鲜行情帧按归属成交重建目标、重新计算对冲。不续发旧未完成开仓目标，当天已选约不会重复开仓；旧价格和 IV 缓存不恢复为新鲜数据。合约集合、条款指纹、日历、来源、时间模式、参数和执行政策必须一致，目前只支持同一执行日的固定资料会话恢复。资料版本改变或跨日要停止重新装配，不能删除旧持仓腿继续运行。

状态文件独立加锁，原生 CTP 另有账户锁。旧格式缺少新日历身份的状态不自动迁移，失败恢复不覆盖原文件。Replay 工程映射不允许 resume，其保存/读回通过不能等同于恢复通过，也不能转作次日实时会话接管文件。

Live 停机撤销自身活动订单并保存已成交仓位，默认不主动平仓。只有显式 `--flatten-day YYYYMMDD --flatten-slot HH:MM` 才在该业务日时点起提交全组合零目标，且仍需新鲜报价和成交反馈。配置清仓日不保证停机前全部成交；应核对最终仓位与活动订单。

### Live 报告

默认写入 `results/option-vega-<mode>-<时间纳秒>/`：

| 文件 / 字段 | 用途 |
| --- | --- |
| `summary.json` | 最终 status、failure、cleanup_errors、通道/参考就绪、模式/时间/日历/来源身份 |
| `events.txt` | Recording 目标请求，或 Model/原生执行回报；启动在装配前失败时可能无此文件 |
| `recorded_requests / entry_audit` | 客户端实际收到的 Recording 请求、选约 TARGET/NO_VALID_ENTRY/ZERO_ALLOCATION/WORKING_ORDERS |
| `frame_audit / late_bar_count / skipped_index_frames` | 帧来源时间、接收期限、CONSUMED/SKIPPED_INDEX、迟到与缺指数情况 |
| `feature_queries` | DFS 缓存、最新业务时间、查询状态；PROJECTED 只说明 Feed 发出，以 frame_audit 确认消费 |
| `strategy_positions / execution_positions / account_positions` | 策略归属仓位、执行账本、物理账户观测；未关联物理账户时为 null |
| `fills_received / order_updates_received / orders_submitted` | 本次归属成交、订单更新及柜台报单计数；目标、接受和成交分别核对 |
| `order_submission_audit` | 原生报单限价与观测盘口诊断，最多 32 条 |
| `replay_test_time_mapping / validation_scope` | 工程映射身份、样例及验收范围 |
| 独立 `state-file` | Model/SimNow 检查点、目标版本、归属、订单、成交去重和政策；不自动放进每次报告目录 |

无 `--require-fills` 时 `acceptance=market_frames`，并不要求非零目标或成交；有该开关时为 `attributed_fills`。还应检查 cleanup_errors、最终柜台活动订单和仓位，不以状态字段代替完整链路证据。

## 6. 代码文件的作用与必要性

### 6.1 是否每个文件都必须

**不是每次运行都需要全部 12 个 Python 文件。** 三个文件实现共用策略核心；其余承担回测入口、在线装配与生命周期、状态保存、工程测试、诊断或报告验收。按当前源码统计，目录内约 2125 行 Python，其中 `vega_signal.py + strategy.py` 为 465 行，约占 22%；其他代码用于输入、运行、执行和验证。行数是本次阅读时的快照，不是后续版本的固定指标。

要区分两个问题：**功能是否需要**，以及**当前入口是否依赖该文件**。例如报单价格诊断不决定策略算法，但当前 SimNow 自动装配它，删掉文件就会启动失败；Recording 不保存检查点，但当前在线装配模块仍导入检查点文件。下面按当前实现说明，不把“本次没有调用”当作“可以直接删除”。

### 6.2 逐文件说明

| 文件 | 主要对象 / 入口 | 具体作用 | 必要性 |
| --- | --- | --- | --- |
| [vega_signal.py](vega_signal.py) | `Contract`、`OptionVegaConfig`、`tau`、`implied_greeks`、`allocate_short`、`filter_hedgeable_options` | 保存条款和配置；计算剩余期限、Black76 认购价格、IV、Delta/Gamma/Vega；按 Delta 档及 Vega 预算分配空头手数，过滤无同月有效期货的候选。纯数值和业务规则，不打开行情/执行连接 | **策略核心**；回测和所有 Live 模式需要 |
| [strategy.py](strategy.py) | `OptionVegaStrategy` | 接收分钟行情；管理每日选约、目标替换、实际成交后 Delta 对冲、到期/时间价值退出、末日清仓及订单异常；检查新鲜报价和场次恢复等待。数学计算调用 `vega_signal.py` | **策略核心**；回测和所有 Live 模式需要 |
| [assembly.py](assembly.py) | `risk_limits`、`assemble_strategy_components` | 共用策略装配：创建手数/名义金额风控、数据绑定和执行路由，注册同一个策略；指数只绑定行情。调用方提供 Feed 与执行积木，它不打开连接 | **共用装配**；当前回测和所有 Live 模式需要，保证两种入口使用相同策略与路由规则 |
| [run_backtest.py](run_backtest.py) | `load_contracts`、`run_case`、`main` | 解析回测参数，读取文件/日历/基础条款，准备期权链和旧持仓腿；检查选约同步和最终清仓时钟；装配原生模拟账户与历史 Runtime，输出报表并核对实际全平 | **文件回测入口**；直接 Live 不依赖它 |
| [run_live.py](run_live.py) | `parse_args`、`prepare_inputs`、`build_runtime`、`describe`、`main` | 解析在线模式、来源、交易日和执行政策；准备 CTP/DolphinDB 行情、独立指数、参考资料与编译日历；选择生命周期/执行模式并生成在线报告 | **在线入口**；所有 Live 模式需要，文件回测不依赖它 |
| [live_runtime.py](live_runtime.py) | `OptionReferences`、`OptionSessionRunner`、`assemble` | 把公共条款快照投影为本策略的认购与同月期货关系；收集/封闭多源分钟帧、查询指数、检查日期与时效、跳过缺指数帧；选择 Recording/Model/CTP 执行并装配状态管理 | **在线运行装配**；所有 Live 模式需要。分钟帧缓冲、执行、归属和持久化基础能力复用公共框架 |
| [recovery.py](recovery.py) | `OptionControl`、`OptionLocalLifecycle`、`OptionCtpLifecycle` | 负责启动就绪、控制线程轮询、参考刷新、行情检查、成交验收、停机报告；Local 处理 Recording/Model，CTP 处理 TD 对账、授权、报单上限、撤单与保存；恢复时核对归属及目标版本 | **在线生命周期**；即使不传 `--resume`，首次 Live 启动与停机也需要。文件名包含 recovery，但职责覆盖整个会话 |
| [checkpoint.py](checkpoint.py) | `OptionVegaCheckpoint` | 保存期权/期货目标、已选约日期、提交/选约计数和清仓状态；恢复时校验数量、合约、日历，清空旧价格与 IV 缓存，并标记等待新鲜行情。公共持久化另存成交、归属及订单 | **Model/SimNow 必需**；新会话也要保存，不只用于重启。Recording 不使用状态保存，但 `live_runtime.py` 顶层导入它，当前仍须随 Live 源码提供 |
| [replay_test.py](replay_test.py) | `ReplayTestTimeMap`、`mapped_bar_feed` | 在完成 CTP Bar 的副本上施加所有合约共用的时间偏移，让 Replay 流推进到指定估值时点；校验源日期/跨日，保存映射样例。文件名含 test，但它是可被入口调用的工程适配器，不是 unittest 文件 | **条件使用**；仅显式 `--replay-test-start-time` 时导入/调用。普通实时和原始时间 Replay 不需要该功能 |
| [execution_audit.py](execution_audit.py) | `AuditedExecutionBackend` | 包装原生执行 Backend，记录实际限价、分钟参考价、已观测盘口及年龄、是否跨价、提交调用结果。原订单发送一次，异常继续抛出；不调整定价或模拟成交 | **诊断功能**；当前 SimNow 无条件装配、没有关闭 CLI 开关，因此现有 SimNow 启动必须保留。回测/Recording/Model 不使用 |
| [acceptance.py](acceptance.py) | `review_recording` | 离线检查真实 Recording 的 summary：日期、模式、编译日历、指数查询、选约同分钟/接收期限，以及客户端实际非零目标。返回 target_chain 或 decision_only 验收结果，不连接网络 | **验收工具**；直接 `run_live.py` 不调用。`scripts.integration.options.recording` 在自动运行或复核已有报告时需要它 |
| [check_backtest_data.py](check_backtest_data.py) | `inspect_file`、`inspect_market`、`main` | 最近新增的只读排错工具：按相同路径规则与分钟标签检查失败时点的期货/指数文件，输出实际路径、时间范围、价格年龄和附近行。用于定位 IM2609 隔夜持仓缺报价问题 | **排错工具**；两个正常运行入口都不导入它。需要查输入覆盖时才单独执行，诊断不等同于回放送达验证 |

### 6.3 不同运行方式实际依赖哪些文件

“必需”表示当前模式会使用；“导入”表示当前装配仍要求文件存在，虽然该模式不启用其功能；“条件”表示显式工程映射才使用；“—”表示该模式的直接入口不依赖。表中只统计 `08_option_vega` 自身文件，公共框架、第三方依赖、原生扩展和数据配置仍需具备。

| 文件 | 文件回测 | Recording | Model | SimNow |
| --- | --- | --- | --- | --- |
| `vega_signal.py` | 必需 | 必需 | 必需 | 必需 |
| `strategy.py` | 必需 | 必需 | 必需 | 必需 |
| `assembly.py` | 必需 | 必需 | 必需 | 必需 |
| `run_backtest.py` | 必需 | — | — | — |
| `run_live.py` | — | 必需 | 必需 | 必需 |
| `live_runtime.py` | — | 必需 | 必需 | 必需 |
| `recovery.py` | — | 必需 | 必需 | 必需 |
| `checkpoint.py` | — | 导入 | 必需 | 必需 |
| `replay_test.py` | — | 条件 | 条件 | 条件 |
| `execution_audit.py` | — | — | — | 必需（自动诊断） |
| `acceptance.py` | — | — | — | — |
| `check_backtest_data.py` | — | — | — | — |

因此，**只运行文件回测时，本目录内的代码依赖为 4 个文件**：`run_backtest.py`、`assembly.py`、`strategy.py`、`vega_signal.py`。这是最小代码依赖说明，不能把这四个文件单独复制到空目录就认为所有依赖已满足。

直接 Live 的基础源码依赖为 Recording/Model 7 个文件、SimNow 8 个文件；启用 Replay 工程映射再增加 `replay_test.py`。使用自动 Recording 运行/复核工具时，再需要本目录的 `acceptance.py` 和目录外 `scripts/integration/options/recording.py`。只读回测数据诊断则单独运行 `check_backtest_data.py`，不依赖策略或执行模块。

### 6.4 调用关系与阅读顺序

下面只画目录内主要调用关系，文件、网络、撮合、订单、归属和公共状态管理实现位于 `bomber/framework/`：

```text
文件回测：
  run_backtest.py
    → assembly.py → strategy.py → vega_signal.py

在线运行：
  run_live.py
    → live_runtime.py → assembly.py → strategy.py → vega_signal.py
        → checkpoint.py          Model/SimNow 状态保存与恢复
        → execution_audit.py     SimNow 自动报单诊断
    → recovery.py                启动、轮询、验收、恢复、停机
    → replay_test.py             仅启用 Replay 工程时间映射

独立工具：
  scripts/integration/options/recording.py → acceptance.py
    自动工具运行时另启动 run_live.py；--summary 时只复核已有报告
  check_backtest_data.py → 公共 dataprep 文件读取与质量校验
```

了解策略先读 `vega_signal.py → strategy.py → assembly.py`；跑回测再读 `run_backtest.py`。跑 Live 再读 `run_live.py → live_runtime.py → recovery.py`；关注状态保存/恢复时读 `checkpoint.py`。其余三个工程/诊断/验收模块按任务阅读，回测排错时单独看 `check_backtest_data.py`。

### 6.5 为什么拆成多个文件，哪些可以精简

算法、策略状态、输入来源、执行装配和会话生命周期有不同的职责。现在的划分让回测与 Live 共用选约/对冲逻辑，并让工程时钟或报单诊断的修改集中在对应模块。Python 并不要求拆成 12 个文件；合并文件可以减少文件数，但不会消除输入校验、分钟同步、归属、保存或停机所需的逻辑。

保留回测与 Live 全部现有能力时，核心、入口、装配和生命周期都有实际调用。工程映射、独立验收和输入检查可按使用范围选择；报单审计若要从 SimNow 中移除，需要同时修改在线装配及对应测试。当前没有发现完全无调用入口或无引用用途的 `.py` 文件。当前用户要求是说明和更新文档，因此本次不删文件，也不改变运行行为。

可以优化的一个明确依赖是：将 `live_runtime.py` 中的 `OptionVegaCheckpoint` 导入移入状态管理分支，Recording 就不再需要仅为导入而带上检查点模块。这属于后续代码精简，不改变 Model/SimNow 新会话仍须保存状态的要求。

`__pycache__/` 是 Python 自动生成的字节码缓存，可重新生成，不属于需要人工维护的策略源码。`results/` 属于运行输出；其中 Model/SimNow 的原状态文件、锁和报表要结合会话判断用途，不能因为它们是输出就当成无用代码统一清理。测试代码主要放在项目 `tests/`，诊断工具的测试为 `tests/test_option_vega_backtest_data.py`。

### 6.6 使用的公共组件

回测和 Live 都由 `assembly.py` 创建策略、数据绑定与执行路由。策略使用 `position()` 和 `strategy_working_quantity()` 查询自身归属成交及在途分配，不以物理账户总仓或其他策略委托计算 Delta。

第三方一对多执行通过 `build_runtime(args, account_routes=...)` 和公共 `AccountExecutionRoute` 装配，策略主账本选择 Model；目前没有对应命令行账户路由参数，也没有新增厂商网络连接器。08 不引用其他 demo 文件。

#### dataprep：输入准备


| 关键接口 | 本例用途 |
| --- | --- |
| input_session / resolve_option_args | 共享来源与读取缓存，解析期权、期货、指数行情及基础条款路径 |
| load_calendar / infer_market_calendar | 读取显式日历或推导全部已知行情日期，为期限判断提供日期集合 |
| load_options_basic / load_cffex_futures | 校验股指认购与真实期货条款，返回规范化静态资料 |
| inventory / BarFileKey / plan_option_chain | 索引三类行情，声明选约及持仓管理所需文件，生成 InputPlan |
| BarReadSpec / prepare_option_chain | 按日盘一分钟统一时间、身份和严格 OHLCV 校验，返回 HistoricalInputBundle |
| add_prepared_bar_source | 注册公共层已准备行情，不重新读取原始文件或重复移动时间 |
| write_input_reports | 输出输入版本、覆盖与问题诊断 |

#### datahub：参考资料查询

| 关键接口 | 本例用途 |
| --- | --- |
| 参考资料来源 | 回测使用文件条款；Live使用DolphinDB条款快照。策略只使用准备后的认购条款、同月期货映射和日历，不查询数据库、连续主力角色或复权价格 |

#### market：真实行情回放

| 关键接口 | 本例用途 |
| --- | --- |
| FileReplayFeed / InstrumentMeta | 注册认购、同月期货与指数真实行情；指数仅提供信号行情元信息 |
| DataType.BAR / MarketStreamBinding | 声明交易腿一分钟回放流；指数不创建执行合约或交易路由 |

#### trader：卖方及对冲执行

| 关键接口 | 本例用途 |
| --- | --- |
| instrument_factory.make_option / make_future / instrument_meta | 构造原生期权、期货合约及对应行情元信息，无例子内转发文件 |
| StrategyTemplate / set_targets | 提交认购和对冲期货的完整目标，包括旧腿零目标 |
| CtpFuturesBasicProfile / NautilusSimExecutionBackend | 模拟账户、手续费、基础保证金及原生合约回测 |
| PositionManager / NetTargetOrderPlanner | 管理实际与在途持仓，把目标转换为净持仓订单 |
| MarketReferencePriceStore / PreTradeRiskManager / RiskLimits | 校验真实报价时效、手数和名义金额，指数没有交易风控路由 |
| DataBinding / ExecutionRoute | 按符号绑定三类信号行情，只给期权与期货创建执行路由 |
| UnifiedStrategyRunner / NautilusMarketFeedAdapter / UnifiedHistoricalRuntime | 串联真实行情、策略、模拟执行与历史回放生命周期 |

## 7. 验证方法与现有证据

具备原生 SDK 的目标环境可运行离线验收；这些命令使用行情/执行夹具，不连接数据库或柜台：

```bash
python -u -m tests.run_cross_section_acceptance \
  --only option-vega --only option-vega-live \
  --only live-common --only attribution --only assembly \
  --only option-calendar --only feature-query-live
```

无原生 SDK 的开发环境可运行：

```bash
python -m unittest tests.test_demo_option_vega tests.test_option_vega_live
python -m tests.run_option_vega_offline
```

第二条替换 SDK 行情实体和未使用的 SDK 桥，实际 Runner、归属、风控、执行状态机与 JSON 持久化仍是项目代码，不能替代原生环境验收。

截至 2026-10-12 的已有证据来自专项交接，**不是本次文档更新新运行的结果**：

| 范围 | 已有结论 | 仍需单独确认 |
| --- | --- | --- |
| REV3 远程离线回归 | 五组选定组共执行 195 项通过；真实指数只读探针正确映射 14:18—14:24 七条记录 | 只读查询不证明原运行帧截止前可见 |
| 实时非映射 Recording | 曾因首行情缺失/同分钟指数缺失失败，保留原始失败证据 | 新版本实时非零目标链；2026-10-13 的交接命令是计划，不能登记为完成 |
| Replay 工程 Recording | 单档 0.25 运行 10 帧，收到 MO2610-C-7300 空 1 手非零请求 | 不代表实时路径通过 |
| Replay 工程 Model | 9 帧、1 次归属模型成交、2 次订单更新，策略及执行仓位 -1；状态读回 generation=14 | 没有非零期货目标；重启恢复未验 |
| Replay 工程原生 SimNow | 首次订单接受后未成交并撤单；随后带价格诊断运行成交：MO2610-C-7300 卖开 1 手，限价 37.2、成交 37.4、柜台单号 12681；策略/执行/账户净仓均 -1，停机活动订单 0，状态读回 generation=15 | 实时非映射原生成交、非零期货对冲成交、重启恢复未验 |
| 9 月 1—9 日原生文件回测 | 用户返回的远程运行成功：7 个交易日、1687 帧、7 次选约、32 笔订单、33 笔模拟成交，最终检查及清理完成 | 原始报表尚未单独读取；9 月 10 日缺报价问题及延长区间仍需核对 |

### 2026-10-12 返回的文件回测成功记录

用户在远程 `uv-nautilus` 环境运行 `run_backtest.py --option-product MO --future-product IM --index-code 000852 --start-day 2026-09-01 --end-day 2026-09-09 --bar-timestamp end --log-level WARNING`。日志报告实际 7 个交易日、105 个认购合约、4 个同月期货；`frames=1687`、`entries=7`、`orders=32`、`fills=33`。回测阶段 75.212 秒，main 总耗时 88.428 秒，报表路径为运行机 `pro/demos/08_option_vega/results/70d4ca7d-e457-4f78-a7f2-0b2a4f7d7bed`。

日志到达最终结果校验完成和 main 完成，按当前入口的检查可推知没有残留策略仓位/在途数量、回报错误且已触发末日清仓；本次尚未单独读取其 `summary.json`、逐笔报表或输入文件，因此不补记收益、指数真实目录或逐笔正确性结论。它验证的是这个区间的文件回放与模拟成交链路；此前 9 月 1—11 日运行在 9 月 10 日缺少 IM2609 新鲜报价，仍需另行定位。

原生成功会话的状态文件 `demos/08_option_vega/results/replay-ctp-20261012-114149.json` 是交接记录中的远程原文件，应保留；本机不假设它存在，也不覆盖或作为新会话文件使用。最新代码交接还记录 PF2026101207 的冻结执行组件包装修复及本地装配回归，与此前版本的测试证据分别解读。

阅读依据与详细记录：

- [文档导航](../../doc/README.md)、[统一策略架构](../../doc/architecture/STRATEGY_ARCHITECTURE.md)、[公共输入/运行设计](../../doc/strategies/DEMOS_INPUT_RUNTIME_DESIGN_2026-10-04.md)。
- [08 回测与 Live 改造](../../doc/HANDOFF/live/HANDOFF_OPTION_VEGA_2026-10-11.md)、[日历、Recording、Replay 及原生成交交接](../../doc/HANDOFF/live/HANDOFF_OPTION_CALENDAR_RECORDING_2026-10-12.md)。
- [声明式指数特征接入](../../doc/HANDOFF/gaps/HANDOFF_GAP08_DECLARATIVE_2026-10-12.md)、[Replay 原生成交证据](../../doc/evidence/live/OPTION_REPLAY_CTP_NATIVE_FILL_20261012.json)。
