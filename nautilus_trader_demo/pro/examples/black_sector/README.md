# 第五类：黑色系次主力收益率，RB 主力执行

当前实现对应原 `example08/09/10` 的共同策略：JM、I、RB 三个品种的**次主力**真实合约 Bar，分别乘各自累计复权因子；同步后计算收益率。JM 和 I 收益率各取 MA30，再取两者均值的 MA15；RB 当前收益率高于该值时目标 `rb_main=+1`，低于时为 `-1`，相等保持原方向。`30/15` 表示同步 Bar 数；输入一分钟 Bar 时不是 30/15 个交易日。

积木位置：

```text
Feather/其他Reader → FileReplayFeed/实时Feed → BlackSectorTargetStrategy
角色表/真实收盘价 → SectorRoleStore ↗       → rb_main逻辑目标
                                                  ↓
                    ScheduledContractResolver → SafeRollCoordinator
                                                  ↓
                       Planner → Risk → NautilusSim/Live Backend
```

- `datahub/sector_roles.py` 通用维护“品种 → 角色 → 真实合约/因子”的因果快照，不包含 JM/I/RB 或 RB 主力常量；未来可由正式 DataHub Provider 产生相同记录。
- `sector_strategy.py` 的 `BlackSectorConfig` 声明本策略的两个领头品种、比较品种、信号角色、执行品种/角色和逻辑目标键。默认值才是 JM/I/RB、次主力信号与 RB 主力执行。
- `local_input.py` 是当前真实 Feather 样本适配器，接收上述配置。它从上一交易日角色表取合约，利用旧、新真实合约在同一来源日的收盘价，分别计算各信号品种的换约因子。缺锚点时不默认因子为 1；本模块不是 DataHub 核心。
- `sector_signal.py` 是可配置两个领头品种与一个比较品种的纯收益率/窗口信号核，不依赖 Bomber、行情文件或下单端；文件名避免遮蔽 Python 标准库 `signal`。
- `sector_strategy.py` 等 JM/I/RB 同时间戳真实 Bar 到齐才决策；只提交逻辑 `rb_main` 目标，不自己生成订单。默认 `submission_delay_bars=0`，信号 Bar 后立即提交目标；统一 Historical Runtime 已先处理该 Bar，故原生 Backend 最早在后续行情事件撮合。若设置为 `1`，策略再等目标 RB 主力**自己的下一根严格晚于信号时刻**的 Bar 才提交，实际最早成交会再晚一根。不能把目标提交 Bar 当成成交 Bar，也不承诺 next open/next close 的精确成交价。
- `run_bar_backtest.py` 装配 CTP Basic Profile、标准 Feed、动态路由、仓位、风控、原生模拟撮合。仅验证基础撮合，不声称已覆盖 CTP 平今/平昨或逐日结算。

先逐阶段验证：

```bash
export PYTHONPATH=.
python tests/run_black_sector.py --stage roles
python tests/run_black_sector.py --stage signal
python tests/run_black_sector.py --stage strategy
```

真实样本回测，在装有 Bomber、pandas、pyarrow 的 `uv-nautilus` 环境运行：

```bash
python -u examples/black_sector/run_bar_backtest.py \
  --start-day 2026-01-05 --end-day 2026-02-26 \
  --bars-dir /workspace/data/dev/kd/intelkit/records/temp \
  --contract-struct /workspace/worker/pj/neutron/tests/temp/role/fut_contract_data.feather \
  --fut-basic /workspace/worker/pj/neutron/tests/temp/role/fut_basic.feather \
  > /tmp/black_sector.log 2>&1
echo "exit_status=$?"
grep -E 'V4正式基础回测|V4目标审计' /tmp/black_sector.log
```

验收需检查 `complete_frames > 0`、目标提交/实际成交的时间先后、只出现 RB 主力订单、换月旧仓先归零再开新仓。默认立即提交时 `signal_ns == target_submission_ns` 是正确的，但真实成交应更晚。`orders/fills` 可能为零，取决于样本内是否完成预热、信号是否变化，以及最后信号后有没有后续行情；不能仅凭进程退出码宣称策略成交已验证。

当前限制：没有在本地 macOS 环境执行原生 Bomber 撮合；真实样本文件位于远程容器，需按上面的顺序在该环境验证。`pcr_cumfactor` 与本地同日收盘价计算的因子还需要独立核对锚点和生产口径，不能直接将外部累计因子套到三个次主力。期货 Basic Profile 的保证金与手续费仅为基础示例配置。
