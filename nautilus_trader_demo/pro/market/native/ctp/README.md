# CTP 原生行情层

- `driver.py`：隔离 Python 适配层与 CTP 驼峰接口；
- `binding/`：独立构建的 `bomber_ctp_md` pybind11 扩展；
- 上层入口：`market.stream.ctp.CtpLiveDataFeed`。

构建和无柜台验证请进入 `binding/`，按其中 README 执行。
