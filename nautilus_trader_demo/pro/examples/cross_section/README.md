# 第二类：多品种截面动量（example03 / example04）

同一个`CrossSectionMomentumStrategy`接两种离线输入：五品种Feather 1分钟Bar，或
五品种CTP一档Quote Tick→已收盘MID 1分钟Bar。策略只在五路**同一事件时间**的Bar
全部到齐时推进一次窗口；不前向填充、不因缺路使用旧Bar，也不在重复Bar上重复决策。

默认配置沿用原示例：RB、RM、SA、CF、M五个合约；收益率回看20根完整同步Bar，
每5个完整同步时点调仓；做多收益率最强、做空最弱，其余三腿目标为0。每条非零腿
以`目标名义金额 / (当前收盘价 × 对应合约乘数)`向下取整，至少1手；一次
`set_targets()`提交五腿完整快照。历史Tick聚合使用一档MID，不使用成交Tick；
Quote没有真实成交量，衍生MID Bar的volume为0。两种输入不要求得到同样的收益或信号，
只要求使用同一策略规则和目标/执行主链。
Tick回测还将原始Quote转发给模拟Backend作为盘口撮合输入；策略仍只接收MID Bar。

## 按阶段验证（均不连接行情/交易网络）

在`pro`目录执行，先设置`export PYTHONPATH=.`。前两步仅用内存Bar和Recording；
后四步自动生成临时的五品种CSV/Feather，Formal阶段使用Nautilus模拟撮合。
Formal测试请分开运行，每次仅创建一个回测引擎。

```bash
python tests/strategies/cross_section/run_test.py --stage 1
python tests/strategies/cross_section/run_test.py --stage 2
python tests/strategies/cross_section/run_replay_test.py --stage bar-feed
python tests/strategies/cross_section/run_replay_test.py --stage tick-feed
python tests/strategies/cross_section/run_replay_test.py --stage bar-formal
python tests/strategies/cross_section/run_replay_test.py --stage tick-formal
```

阶段1检查缺路、重复、时间回退；阶段2检查20根预热、每5帧调仓、排序及五腿
完整目标；两个feed阶段分别验证无标的列的Feather解析和无Volume列的CTP报价解析、
MID半跳精度与收盘后才发布；两个formal阶段验证两种输入共用策略且进入统一模拟链。

## 用实际历史文件运行

`--data-dir`下应有`RB/ RM/ SA/ CF/ M/`五个子目录；每个子目录可放多个同品种、
同格式文件。脚本按原示例固定合约：`rb2610.SHFE`、`RM609.CZCE`、
`SA609.CZCE`、`CF609.CZCE`、`m2609.DCE`。文件内合约应与目录对应。

- Bar：各目录放`*.feather`，列至少包含`datetime,open,high,low,close,volume`。
  无时区`datetime`按`Asia/Shanghai`解释；文件本身无需symbol/exchange列。
- Tick：各目录放`*.csv`，列至少包含`TradingDay,InstrumentID,UpdateTime,`
  `UpdateMillisec,BidPrice1,BidVolume1,AskPrice1,AskVolume1`。夜盘自然日若与
  TradingDay不同，显式传`--night-session-action-day YYYYMMDD`。

```bash
python examples/cross_section/run_backtest.py --source bar --data-dir /path/to/bar_data
python examples/cross_section/run_backtest.py --source tick --data-dir /path/to/tick_data
```

正式数据接入前需核对文件时间、合约代码、价格精度和交易规则。本示例的激活/到期
时间、12%保证金、固定每手手续费及单笔250,000元风控阈值是装配用近似值，
**不是正式CTP交易规则或实盘参数**。`CtpFuturesBasicProfile`使用净持仓，
未模拟今昨仓和平今/平昨。多腿目标同批提交，不等于交易所层面的原子成交；
跨品种成交风险需要单独评估。停机不会隐式平仓。
