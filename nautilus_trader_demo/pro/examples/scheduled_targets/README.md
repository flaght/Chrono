# 第六类：外部目标计划驱动

核心链路：

```text
CSV/其他来源 → TargetScheduleStore → 独立时钟 → ScheduledTargetStrategy
  → TargetPortfolio(REPLACE) → PortfolioCoordinator → ExecutionRoute
  → Simulation 或 Live Backend
```

`datahub/target_schedule.py`只定义格式无关的计划和因果读取；CSV 读取仅在本目录
`local_input.py`，兼容`timestamp,instrument_id,instrument_type,target_qty`旧文件，
也接受`target_key`列。策略既不读取文件，也不通过 Bar 回调判断时间。每个时点的一组行
构成完整组合：数量 0 是明确平仓；下一时点未列出的旧目标会被 REPLACE 移除，
Runner 向原执行客户端发送显式 0。所有交易目标必须配置 ExecutionRoute；指数若只
用于研究/估值，不能误配为下单目标。
`mixed_contract_fixture.csv`仅验证期货、期权和指数记录可构成同一计划，**不是**
真实可交易合约表，也不用于绩效计算。

逐步验证：

```bash
export PYTHONPATH=.
python tests/run_scheduled_targets.py --stage plan
python tests/run_scheduled_targets.py --stage strategy
python tests/run_scheduled_targets.py --stage runner
python tests/run_scheduled_targets.py --stage replay
```

前两阶段分别检查计划因果语义、策略精确触发和防重；`runner`阶段使用记录型客户端，
**不撮合、不真实下单**；`replay`阶段检查同一时间戳先行情后计划及末尾未到达计划。

真实 CTP 期货 Bar 与 Nautilus/Bomber 撮合（需用户服务器上的两份 Feather 文件）：

```bash
python -u examples/scheduled_targets/run_ctp_bar_backtest.py \
  --bars-dir /workspace/data/dev/kd/intelkit/records/raw_data/cn_futures/20260728 \
  --targets examples/scheduled_targets/positions_20260728.csv \
  > /tmp/scheduled_ctp.log 2>&1
echo "exit_status=$?"
grep -E 'S5正式基础回测|S5调度审计|Traceback|Error' /tmp/scheduled_ctp.log
```

样本计划含 09:40、09:45 两个时点，两个真实合约 `rb2704.SHFE`、`SA703.CZCE`。
合约费用、保证金和起始资金沿用 `cross_section` 的基础示例参数；正式绩效评估前
仍需历史规则核准。若文件不覆盖计划时点，审计显示 `NOT_REACHED`，不伪造订单。
正式模拟Backend仍由市场事件推进时钟：计划落在两根Bar之间时，目标虽按计划时点
提交，模拟撮合时钟到下一根Bar才继续推进。需要精确到无Bar时间点的成交/超时规则时，
应补充Backend时间事件，不应将此基础回测当作毫秒级执行验证。

在线侧可用 `ManualClockFeed.emit_time()` 接入外部日历/调度器，与实时 Market Feed
并列配置；此处只提供时钟端口，尚未实现生产用持久化定时服务、重启补偿与活动订单
对账。期货+期权+指数混合计划可以载入并验证目标语义，但缺真实期权合约主数据、
分钟行情及撮合配置时，不得宣称其正式回测或实盘可用。
