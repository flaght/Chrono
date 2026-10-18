# CTP SimNow＋Binance DEMO 联合入口

GAP-03更新：联合入口默认DolphinDB参考资料，文件资料须显式`--reference-source file --allow-file-reference-test`。expected-source-day在每次刷新仍核验；来源年龄／成功观测／单轮刷新上限默认14自然日／30秒／30秒，可用reference-max-source-age-days、reference-max-observation-seconds、reference-max-refresh-duration-seconds调整。数据库失效不回退旧文件；来源／时效异常通过既有共同门控限制执行。此处CLI和参考公共代码已变化，须按 [GAP-03交接](../../../doc/HANDOFF/HANDOFF_GAP03_2026-10-09.md)复测joint；此前24项通过是旧版证据，不覆盖本次修改。

PF2026100905。2026-10-09远程joint 24项全部通过（测试0.043秒、模块退出0/1.114秒，报告main-ema-acceptance-1791511018663080359）；controlled/assembly/dispatch/entry、Binance活动订单及Live Backend E9全部通过，E9c复测加载摘要与交付版一致。首期代码与计划内离线回归完成，已通过组无需重跑；当前双端柜台验收待完成，GAP-10尚未关闭。证据及剩余边界见 [交接末尾](../../../doc/HANDOFF/HANDOFF_GAP10_2026-10-09.md)。以下保留首次验收命令：

```bash
python -u -m tests.run_main_ema_acceptance --only joint
```

通过后复测共享 Driver 的受控执行及组装链：

```bash
python -u -m tests.run_main_ema_acceptance --only controlled --only assembly --only dispatch --only entry
python -u -m tests.run_live_backend
python -u -m tests.run_p4_binance_orders
```

首期为固定真实合约、CTP SimNow SHFE/INE CNY期货 LIMIT/GFD，与 Binance DEMO USDT永续单向账户 MARKET。联合策略 `JointTargetProbe` 一次输出两个明确目标；它用于验收路由和成交归属，不提供跨市场交易信号或收益结论。停机撤单并保留成交仓位。跨端非原子，第一端成交后第二端失败可以留下敞口；政策是整体 HALTED，查询及人工处理后分别调用恢复、授权和剩余目标规划，没有自动补偿或换端重发。

配置沿用 `.env` 的 CTP_BROKER_ID=9999、CTP_ACCOUNT_ID、CTP_PASSWORD、CTP_TD_ADDRESS、CTP_MD_ADDRESS，以及可选认证配置。BN 需要 BINANCE_DEMO_API_KEY、BINANCE_DEMO_API_SECRET 和新增 BINANCE_DEMO_ACCOUNT_OWNER_ID：最后一项是稳定的物理账户部署别名，不能用 client_id 或 API key 代替，同账户全部 key 必须配置同一别名。HTTP接口没有在这里核验 UID，别名正确性由部署配置保证。

同机协作进程使用统一物理账户 flock，CTP 兼容已有入口的两套旧锁。没有跨主机 fencing；其他程序、未接入锁的旧 BN 报单入口和人工交易不受此锁保护，首期部署须只保留本入口为写入者。不同账户的目标、归属及在途量分别保存，不能互相净额。

先执行双端只读。预期日期须以本次柜台和参考资料核验为准；下面日期和预算仅示范参数格式，不代表当前账户事实：

```bash
python -u -m scripts.integration.ctp_binance.run \
  --connect --mode readonly --product CU \
  --reference-source dolphindb --reference-database dfs://bomber_daily \
  --expected-trading-day 20261009 --expected-source-day 20261008 \
  --symbol BTCUSDT --ctp-target 1 --bn-target 0.001 \
  --ctp-max-notional 600000 --bn-max-notional 200
```

只读也要求双端全账户空仓、无普通或Algo活动订单。GAP-01已有CU仓位不能由本入口接管，不为满足准入而自动平仓。资金和仓位直接权威查询，缺字段、HTTP错误、BN对冲模式或未知活动订单均拒绝；没有实时目标发送和报单授权。报告中的 `proposed_targets` 只是参数预览，`passed` 在该模式下仅表示本次只读与清理通过。

完成当前只读及离线验收后，在用户明确授权联合模拟订单的范围内，将上面的模式改为 orders，并追加：

```bash
--enable-orders --confirm-simnow --confirm-demo \
--seconds 60 --state-file tests/results/joint-new-session.json
```

状态文件必须不存在。授权前等待双端实时行情和交易会话就绪，核验目标精度及名义额；每端一笔会话上限。下单模式只有双端实际成交回调、完整目标仓位、权威仓位无归属差额、活动订单清零及清理成功才通过。只有单端成交或超时撤单记失败，不自动重试目标，不自动平已成交仓。

输出位于 tests/results/ctp-binance-joint-*：summary.json 保存双端请求审计、订单/成交回调、归属/账户/未归属/在途量、最终CTP总仓及双端活动订单查询；events.txt 保存回调。集中离线入口另保存源码 SHA-256 与模块日志。同步文件及恢复限制见 [GAP-10交接](../../../doc/HANDOFF/HANDOFF_GAP10_2026-10-09.md)。

CTP写前及回报后保存包含双端订单状态机的检查点；BN发送与原生Gateway订单ID之间尚无通用写前恢复协议。本入口拒绝旧状态文件。公共组件支持相同客户端、已知活动订单完整一致的保守恢复；柜台消失订单、漏成交或跨交易日仍按 GAP-06 闭闸，不能将本地文件直接当作重启授权。
