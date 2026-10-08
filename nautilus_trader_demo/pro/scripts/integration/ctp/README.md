# CTP 通道联调

本目录放置独立于策略的 CTP 接入与联调入口。TD 认证、登录、结算确认、权威查询及订单协议实现复用 `trader/execution/ctp`；MD 和行情转换复用 `market/stream/ctp`。

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

三项查询都完整完成才算本模块通过。净仓结果不包含完整今昨仓、成本或双向总仓信息，不能单凭净仓为零认定账户完全无仓；报单接入仍须完成相应的权威持仓明细与活动订单核对。

旧 `examples/single_ema/ctp_td_readonly.py` 仅保留兼容转发，新命令和新引用使用本目录实现。迁移烟测为 `python -u tests/run_ctp_td_readonly.py`，柜台结果仍由用户在远程 Linux 环境提供。
