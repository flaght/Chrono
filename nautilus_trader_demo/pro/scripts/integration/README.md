# 通道接入与联调

本目录提供独立于具体策略的通道运行入口，覆盖连接检查、账户查询、受控订单验证、对账及恢复联调。各入口只编排公共组件，不复制策略公式或底层交易协议实现。

| 子目录 | 通道职责 | 当前入口 |
| --- | --- | --- |
| [ctp](ctp/README.md) | CTP MD／TD 连接、查询与执行联调 | `ctp/td_readonly.py`，交易端只读查询 |
| [binance](binance/README.md) | Binance 行情、账户与执行联调 | 尚未迁入 |

从项目根目录使用模块方式运行 CTP 只读入口：

```bash
python -u -m scripts.integration.ctp.td_readonly --connect --timeout 30
```

该入口始终禁单；`--connect` 只允许建立柜台连接。其他入口迁入后分别说明其网络行为、查询范围、报单能力及所需参数，不能根据同目录其他程序的行为推断授权。

凭据从项目 `.env` 或既有环境变量加载，不写入源码、文档或命令参数。联调结果按入口与环境分别验收；无网络测试通过不代替当前柜台验证。

迁移无网络烟测位于 `tests/run_ctp_td_readonly.py`：

```bash
python -u tests/run_ctp_td_readonly.py
```

烟测使用假 API 和假凭据，核对配置路径、旧入口转发、登录／结算确认、三项查询及禁单，完成后再进行柜台只读联调。
