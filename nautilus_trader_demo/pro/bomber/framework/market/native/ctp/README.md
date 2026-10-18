# CTP 原生接口层

- `driver.py`：隔离 Python 适配层与 CTP 驼峰接口；
- `binding/`：`bomber-ctp` 发行包，包含 `bomber_ctp_md` 和 `bomber_ctp_td` 两个 pybind11 扩展；
- 上层入口：`market.stream.ctp.CtpLiveDataFeed`。

构建和无柜台验证请进入 `binding/`，按其中 README 执行。
