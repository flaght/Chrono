# Binance 通道联调

本目录用于 Binance 行情、账户查询、受控订单验证、对账与恢复联调入口，独立于具体策略。行情能力复用 `market/stream/bn`，交易能力复用 `trader/execution/live`。

当前仅建立目录归属说明，尚未迁入可执行入口。后续先核对旧代码及测试，再分别迁移只读查询和受控订单入口；每个程序明确标注 DEMO／LIVE 范围与授权方式，不能将历史日志或其他通道的验证作为本次 Binance 验收。

2026-10-09：新增独立的 [CTP＋Binance联合入口](../ctp_binance/README.md)，复用Binance DEMO执行接入并新增直接HTTP全账户仓位/资金读取。本目录旧入口迁移仍待进行；新联合源码尚未运行，不据此更新DEMO柜台验收结论。
