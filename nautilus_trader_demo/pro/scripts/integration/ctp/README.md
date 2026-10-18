# CTP 通道联调

本目录放置独立于策略的 CTP 接入与联调入口。TD 认证、登录、结算确认、权威查询及订单协议实现复用 `bomber/framework/trader/execution/ctp`；MD 和行情转换复用 `bomber/framework/market/stream/ctp`。

当前入口为 `td_readonly.py`：登录和结算确认完成后，依次查询净仓、资金及全账户活动订单，最后退出会话。它不提供报单开关，原生 Driver 始终保持禁单。

在已激活的运行环境、项目根目录执行：

```bash
python -u -m scripts.integration.ctp.td_readonly --connect --timeout 30
```

| 参数 | 含义 |
| --- | --- |
| `--connect` | 显式允许柜台连接；不授权订单 |
| `--timeout` | 每项连接／查询操作的超时秒数，默认 15；不是整个程序的总时长 |

配置加载位置固定为项目根 `.env`，已有环境变量优先。必需变量为 `CTP_TD_ADDRESS`、`CTP_BROKER_ID`、`CTP_ACCOUNT_ID`、`CTP_PASSWORD`；沿用既有 `CTP_APP_ID`、`CTP_AUTH_CODE`、`CTP_TD_FLOW_PATH` 和 `CTP_PRODUCTION_MODE` 配置。该入口沿用指定的 CTP 前置配置，不自行选择或认证某一套 SimNow 环境。

公共依赖使用 `bomber.framework` 导入。开发模式须同步项目的 `bomber/__init__.py` 加载文件与 `bomber/framework`，在项目根目录以模块方式运行，复用已安装的 Bomber 内核；发布模式须安装包含 framework 的 Bomber。入口不修改 Python 搜索路径；`PROJECT_ROOT` 仅用于定位项目 `.env`。

三项查询都完整完成才算本模块通过。净仓结果不包含完整今昨仓、成本或双向总仓信息，不能单凭净仓为零认定账户完全无仓；报单接入仍须完成相应的权威持仓明细与活动订单核对。

新命令、新引用及烟测均直接使用本目录实现，运行环境不需要旧 examples 目录。源码中的旧 `examples/single_ema/ctp_td_readonly.py` 仅供尚未更新命令的调用方转发，不属于新入口的依赖。开发烟测为 `python -u -m tests.run_ctp_td_readonly`，柜台结果仍由用户在远程 Linux 环境提供。

## 行情分层验证与本次证据

MD分层探针复用 `tests/run_ctp_live.py`，不连接TD、不发单。账户配置仍来自环境变量／根目录.env，已有环境变量优先：

```bash
CTP_SYMBOL=rb2701 CTP_CONNECT_TIMEOUT=30 CTP_MARKET_TIMEOUT=60 CTP_MARKET_SAMPLE_SECONDS=10 \
python -u -m tests.run_ctp_live --stage 4

CTP_SYMBOL=rb2701 CTP_EXCHANGE=SHFE CTP_PRICE_INCREMENT=1 CTP_MULTIPLIER=10 \
CTP_CONNECT_TIMEOUT=30 CTP_MARKET_TIMEOUT=60 \
python -u -m tests.run_ctp_live --stage 5
```

stage 4检查前置连接、登录、服务器订阅响应及原始Tick；stage 5检查标准QuoteTick与累计成交量增量生成的TradeTick。收到旧交易日Tick也可能通过这两项，不能据此认定行情适用于当前实时策略。

2026-10-08用户远程验证：第二套MD `tcp://182.254.243.31:40011`、production_mode=true，stage 4及5通过；原始TradingDay=20260930、ActionDay=20260929，10秒采样13条Tick。第二套TD `tcp://182.254.243.31:40001`登录、结算确认与三类查询通过，净仓合约0、CNY资金快照、活动订单0。本次TD输出没有交易日，不推断与MD日一致；未执行订单。第一套当前策略Recording与受控柜台成交仍待验收，详情见[01接入验收](../../../demos/01_main_ema/SIMNOW_ACCEPTANCE.md)。

随后用户选择继续第二套完整链路，PF2026100805新增01显式replay工程联调装配。Tick保留原始行情时间，Bar采用实际接收时间；MD原始交易日按参数固定，TD固定为本次登录日，参考资料按当前TD会话／可用时间查询，默认实时模式保持原门控。代码完成待远程entry34项及受控客户端／行情闸门回归，接着第二套Recording与受控模拟订单；同步文件和三步命令见[01第二套完整链路](../../../demos/01_main_ema/SIMNOW_ACCEPTANCE.md#第二套完整链路pf2026100805)。

最新2026-10-08证据：夹具修正版entry34/34及controlled／health全部通过；真实第二套Recording已通过，832条Tick、9根有效Bar、LONG 1手目标，实际MD／TD日20260930、角色／因子日2026-09-29、rb2701.SHFE。停机总仓空、活动订单0、清理无错误，报单0；完整报告为远程 `demos/01_main_ema/results/simnow-1791449059995543664`。下一步是受控模拟订单，仍未登记本次策略柜台成交；无需重复未改动的离线组或Recording。

随后PF2026100806按用户要求将01参考适配及运行器迁入公共dataprep/live_role.py、trader/live_roles.py，demo删除live_references.py／live_runner.py，仅run_live.py组装实盘。以上34项与Recording为迁移前证据；当前先整组同步7文件、删除远程旧文件，复验references／routing／ctp／entry，再继续受控模拟订单。只读探针及原始MD模块没有修改，无需重跑。具体见[单入口迁移](../../../demos/01_main_ema/SIMNOW_ACCEPTANCE.md#单实盘入口迁移pf2026100806)。

PF2026100806最新远程修正版报告main-ema-acceptance-1791458768127640128：references30/30、routing10/10、CTP7/7、entry35/35全部通过，共82项。迁移离线验收完成，下一步是当前单run_live的第二套受控模拟订单；仍未取得本次真实柜台成交证据，首轮假API失败及修正记录见上述迁移说明。已通过且源码未变的模块无需重复。

## 独立处理既有一手模拟仓位（PF2026100808）

2026-10-08两次新EMA Recording启动均被第二套已有rb2701.SHFE空仓1手拒绝；报单0、活动订单0，重跑不会自动清仓。新入口`close_existing.py`独立于策略，整理旧单笔探针的平仓保护，默认只读检查，不依赖examples。原EMA空仓启动和停机不自动平仓规则保留。

原生TD绑定此前没有传出今昨字段，本轮新增PositionDate、HedgeFlag、TodayPosition、YdPosition回调映射；公共transport新增query_position_details，只返回经验证的不可变持仓字段，不泄露账户身份。PositionDate编码来自本项目CTP头文件：1=今日、2=历史。不能依据自然日或旧日志猜测；字段缺失会拒绝提交，要求检查本轮扩展部署。

同步以下5个文件到远程pro的相同路径：

| 文件 | 变化 |
| --- | --- |
| bomber/framework/market/native/ctp/binding/src/bomber_ctp_td.cpp | 新增四个持仓回调字段；需重编扩展 |
| bomber/framework/trader/execution/ctp/td_transport.py | 新增权威持仓明细只读方法 |
| scripts/integration/ctp/close_existing.py | 新增独立一手平仓／默认预检入口 |
| tests/test_ctp_close_existing.py | 新增12项假API验收，包含今昨／方向及异常子场景 |
| tests/run_main_ema_acceptance.py | 新增close-existing模块及源码部署检查 |

Linux uv-nautilus环境、远程pro根目录先执行：

```bash
python -u -m tests.run_main_ema_acceptance --only td --only close-existing
```

2026-10-08首轮远程结果中，`td`原传输回归退出0；`close-existing`的12个测试方法有9个通过，3个直接查询传输层的方法遗漏`activate()`。现已仅修正测试初始化，公共查询就绪守卫保留。远程同步新版`tests/test_ctp_close_existing.py`后只需复测失败模块：

```bash
python -u -m tests.run_main_ema_acceptance --only close-existing
```

复测报告`tests/results/main-ema-acceptance-1791467256985484635`显示新12项全部通过，模块退出0；部署核对中修正后测试SHA-256为`aff56fd79e132adbae2b568e8cdbcbb30d13d6bb573929f3d085def921c70313`。原TD模块此前已退出0，PF07五组无需重复。本机未执行Python；假API仍不能验证真实扩展字段。先确认远程绑定源码与本轮一致：

```bash
sha256sum bomber/framework/market/native/ctp/binding/src/bomber_ctp_td.cpp
# 应为 39941826fc53cecc956482e853893b4b0697c309da3a5209462945fa4094304a
```

然后重编安装现有绑定：

```bash
uv pip install --reinstall --no-cache ./bomber/framework/market/native/ctp/binding
python -u bomber/framework/market/native/ctp/binding/smoke_test_td.py
```

此构建同时生成现有MD／TD扩展，SDK、头文件及动态库不改变，既有凭据仍从环境／根.env读取。smoke只检查扩展方法，不代表账户查询或成交通过。设置本次第二套TD后，默认无单预检：

```bash
export CTP_TD_ADDRESS=tcp://182.254.243.31:40001
export CTP_PRODUCTION_MODE=true

python -u -m scripts.integration.ctp.close_existing \
  --connect --symbol rb2701 --exchange SHFE \
  --price-increment 1 --multiplier 10
```

输出实际TD交易日、全账户多空总仓、今昨明细、需要的买／卖方向及CLOSE_TODAY／CLOSE_YESTERDAY。成功状态为prechecked、orders_submitted=0，仓位保留。仅接受全账户目标合约恰好1手单向投机仓、CNY可用资金、无其他仓位及活动订单；无仓／双向／多手／未知日期或套保类别均拒绝。

2026-10-08用户远程反馈：原生TD smoke显示`bomber_ctp_td methods OK (no network, no orders)`。真实第二套只读预检返回TD交易日20260930、rb2701.SHFE多0／空1手，明细PositionDate=1、TodayPosition=1、YdPosition=0、HedgeFlag=1；计划BUY CLOSE_TODAY，`status=prechecked`、报单0、活动订单0、清理错误为空。真实回报证明当前加载的扩展已传出必要字段；反馈未包含编译输出或源码哈希。此时仓位尚未平掉。

预检及字段核对完成后，用当次盘口和涨跌停范围核对数值限价；获取当次第二套MD原始报价：

```bash
export CTP_MD_ADDRESS=tcp://182.254.243.31:40011
CTP_SYMBOL=rb2701 CTP_CONNECT_TIMEOUT=30 CTP_MARKET_TIMEOUT=60 CTP_MARKET_SAMPLE_SECONDS=10 \
  python -u -m tests.run_ctp_live --stage 4
```

确认MD原始TradingDay仍与本次TD交易日一致，使用当次报价的买一／卖一及涨跌停范围核对数值限价；买入平空仓通常参考卖一价，须由操作者选定本次数值。工具检查限价为正、步长整数倍及名义金额不超过默认50000；涨跌停／盘口由用户核对，不采用文档的历史报价。显式发送需要同时加submit与confirm-simnow，并指定方向、TD日和限价。例如预检仍确认rb2701空仓1手且TD=20260930时：

```bash
# CTP_CLOSE_PRICE须由用户先设置为本次已核对的数字限价。
python -u -m scripts.integration.ctp.close_existing \
  --connect --symbol rb2701 --exchange SHFE \
  --side BUY --expected-trading-day 20260930 \
  --price-increment 1 --multiplier 10 --price "$CTP_CLOSE_PRICE" \
  --submit --confirm-simnow
```

这一条会向第二套柜台发送模拟平仓单。工具从柜台PositionDate选择平今或平昨，报单前重新核对资料与会话，reduce_only=True，原生Driver上限1笔，不发送开仓单。等待默认8秒，未成交则撤销本工具委托并查询；无完整成交不会自动重发或宣布通过。停机输出最终总仓、活动订单、实际报单数和清理错误；只有完整一手成交、全账户总仓空且活动订单0才为passed。失败时先只读核对，不继续EMA。

首笔远程实测：第二套MD的TradingDay=20260930，采样末买一3117、卖一3118、涨跌停[2891,3326]；用户输入买入平今限价3071。柜台ACCEPTED后约8秒CANCELED，成交0；最终`status=failed`、报单1、仍有空仓1手、活动订单0、清理错误空。3071低于当次卖一3118，属于未能按盘口成交的限价。下一步重新获取当次行情，使用最新卖一及涨跌停核对新的买入限价；重新执行工具时仍需显式参数，工具会重新查TD日及仓位。若再次未成交，先查看回报和新盘口，不连续盲目提价或重发。

第二笔远程实测：用户输入买入平今限价3072，反馈没有新MD报价；柜台仍ACCEPTED后约8秒CANCELED，成交0，最终多0／空1手、活动订单0、清理错误空。3072比上一份可见卖一3118低46点。`RuntimeError: 未取得完整平仓成交`是工具对未成交的预期失败退出。当时停止重复提交，重新采样行情并核对当前卖一与TD交易日；买入限价至少触及当前卖一才有可能立即成交。

第三笔远程实测：用户重新采样MD，TradingDay=20260930、买一3114／卖一3115、涨跌停[2891,3326]；随后提交BUY CLOSE_TODAY一手限价3115。柜台回报ACCEPTED、FILLED，成交1手、成交价3115；工具`status=passed`、本次报单1、最终总仓`{}`、活动订单0、清理错误空。既有空仓已归零，当前应按01接入验收继续Recording。此工具成交不能代替EMA策略成交验收。

平仓passed后再重跑01的原Recording命令。完整Recording与本次EMA柜台成交仍须分别验收；这个手工平仓工具成交不能替代策略交易验证。源码及原生扩展回退需整组恢复；不修改或恢复任何柜台仓位。公共变化已在当前对话前后告知，登记[PF2026100808](../../../doc/HANDOFF/HANDOFF_PARALLEL_BACKTEST_LIVE_2026-10-07.md#pf2026100808-独立simnow既有一手仓位处理工具)，跨任务待同步。
