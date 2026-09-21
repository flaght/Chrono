# bomber_ctp_md

这是从仓库内 `3rd/vnpy_ctp` 抽取的**仅行情** pybind11 绑定。Python
模块名为 `bomber_ctp_md`，不会安装或导入 `vnpy`、`vnpy_ctp`，也不包含交易
接口。抽取源码保留原 MIT 许可证；CTP SDK 本身的授权条件仍以供应商为准。

当前 Demo 只为实际部署使用的 Linux x86-64 环境提供构建配置。

## 1. 确认运行环境

```bash
cd /workspace/worker/pj/Chrono/nautilus_trader_demo/pro/market/native/ctp/binding
python - <<'PY'
import platform, struct, sys
print("python:", sys.version)
print("machine:", platform.machine())
print("pointer_bits:", struct.calcsize("P") * 8)
PY

file ../../../../3rd/vnpy_ctp/vnpy_ctp/api/libthostmduserapi_se.so
```

预期是 Python 3.12、`x86_64`、64 位，CTP 动态库为 ELF x86-64。

## 2. 编译安装

```bash
uv pip install .
```

如果想确保没有使用旧构建缓存：

```bash
uv pip install --reinstall --no-cache .
```

## 3. 无网络 smoke test

```bash
python smoke_test.py
```

这个测试只验证扩展导入、CTP 动态库加载、API 版本、对象创建/释放和未初始化
调用保护，不会连接行情前置，也不需要账号密码。

构建会复用 `/pro/3rd/vnpy_ctp/vnpy_ctp/api` 中已经存在的官方 CTP 头文件和
`libthostmduserapi_se.so`。后续可把官方 SDK 放到独立的 `vendor/ctp` 目录，
并相应修改 `ctp_api_dir`，避免目录名继续带有 vn.py。

相较原绑定，此副本已经做了三项最低限度修正：任务指针默认初始化、固定长度
CTP 字符数组的有界复制、停止标志/队列终止的线程同步。它仍是一个简化候选，
正式生产前应继续增加有界原生队列以及停止时未处理任务的释放策略。
