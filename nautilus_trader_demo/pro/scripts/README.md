# 脚本目录职责

本目录放置可独立运行的工程工具和通道联调入口，按用途划分。工程维护工具负责源码同步与发布准备，通道联调入口负责验证交易系统连接、账户查询和执行链路。

```text
scripts/
├── sync-bomber-framework.py
├── README.md
└── integration/
    ├── README.md
    ├── ctp/
    │   ├── README.md
    │   └── td_readonly.py
    └── binance/
        └── README.md
```

| 位置 | 职责 | 当前内容 |
| --- | --- | --- |
| scripts 根目录 | 工程维护与框架发布准备 | `sync-bomber-framework.py` 将 `bomber/framework` 原样同步到目标项目，保留构建资源并检查旧导入；默认只报告差异，`--apply` 构建暂存副本 |
| [integration](integration/README.md) | 独立于策略的通道接入与联调 | 组织 CTP、Binance 的连接、查询、执行验证及恢复联调入口 |
| [integration/ctp](integration/ctp/README.md) | CTP 行情与交易通道联调 | 当前已有 TD 只读入口：登录、结算确认、仓位／资金／活动订单查询 |
| [integration/binance](integration/binance/README.md) | Binance 通道联调 | 当前已建立归属说明，入口尚未迁入 |

`scripts/integration` 编排现有公共能力；连接、规划、风控、回报与恢复实现归 `trader`，行情实现归 `market`。具体策略及回测／实盘装配归 `demos`，自动断言与回归测试归 `tests`。新代码不依赖旧 `examples` 入口；迁移期间旧入口可以转发到新实现。

发布工具与柜台联调分别运行。执行 `sync-bomber-framework.py` 不代表连接柜台；运行 `integration` 入口也不承担框架打包或发布。
