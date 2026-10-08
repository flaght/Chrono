# 参考数据库只读核验

本目录核验参考资料读取和字段契约，不连接 CTP MD／TD，也不报单。公共实现位于 [dataprep sources](../../../bomber/framework/dataprep/sources/README.md)，入口加载项目 .env、接收参数并展示结果。

远程 Linux uv-nautilus 环境的 pro 根目录先执行无网络增量：

```bash
python -u -m tests.run_main_ema_acceptance \
  --only references --only reference-regression --only entry
```

通过后，沿用项目 .env 或环境变量中的 DDB_HOST、DDB_PORT、DDB_USERNAME、DDB_PASSWORD。以下交易日须替换为实际已核对的 TD 交易日：

```bash
python -u -m scripts.integration.reference_data.readonly \
  --connect --trading-day 20261008 --products RB \
  --database dfs://bomber_daily --option-product MO \
  --report tests/results/reference-data-20261008.json
```

去掉 --option-product 时仅核验期货链。--products 可传多个品种，所有请求品种须有同一来源日的角色、本次交易日累计因子及有效合约条款。缺失、冲突、未来发布时间或读取中变化都会失败，不将旧因子日期换成当天。

结果包含来源日、真实主力、累计因子、乘数、价位、资料指纹及可选期权核验，不输出密码。若数据库最新因子只有2026-09-30，而核验交易日为2026-10-08，仍须上游补齐2026-10-08的真实因子。

随后验证当前策略实时 Recording：

```bash
python -u -m demos.01_main_ema.run_live \
  --connect --mode recording --product RB --seconds 600 \
  --reference-source dolphindb --reference-database dfs://bomber_daily
```

该命令连接 SimNow MD／TD并核验账户，Recording 不报单。数据源选择只影响参考资料，实时行情仍来自 CTP，交易日仍由 TD 确认。数据库模式不要求角色、因子或条款文件路径。完整交易授权及停机边界按 [01 接入验收](../../../demos/01_main_ema/SIMNOW_ACCEPTANCE.md) 执行。
