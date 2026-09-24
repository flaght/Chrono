# bomber-ctp

本目录构建两个项目内的 pybind11 模块：`bomber_ctp_md` 提供行情 API，
`bomber_ctp_td` 提供 TraderApi。TD 绑定由项目直接实现，仅使用本目录中的
CTP SDK 头文件与动态库，不编译或导入 vn.py / vnpy_ctp 的交易绑定源码。
TD 接口声明在 `src/bomber_ctp_td.hpp`，实现与 Python 注册在
`src/bomber_ctp_td.cpp`。
MD 绑定仍保留较早迁入的生成代码，后续可单独重写。CTP SDK 的授权条件
以供应商为准。

发行包名是 `bomber-ctp`；安装一次会生成两个独立的 Python 扩展模块
`bomber_ctp_md` 和 `bomber_ctp_td`。

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

file libthostmduserapi_se.so
file libthosttraderapi_se.so
```

预期是 Python 3.12、`x86_64`、64 位，CTP 动态库为 ELF x86-64。

## 2. 编译安装

`0.1.0` 版本的发行包名为 `bomber-ctp-md`。同步本目录的新配置后，
先移除旧发行包，再安装包含 MD 和 TD 的 `bomber-ctp`：

```bash
uv pip uninstall bomber-ctp-md
uv pip install .
```

如果想确保没有使用旧构建缓存：

```bash
uv pip install --reinstall --no-cache .
```

## 3. 无网络 smoke test

```bash
python smoke_test.py
python smoke_test_td.py
```

两个测试只验证扩展导入、CTP 动态库加载和基本 API 能力，不连接任何前置，
不需要账号密码，也不会报单。

构建只读取本目录 `include/ctp` 中的官方 CTP 头文件和上述两份 SDK 动态库。
TD 自有绑定只映射当前交易传输层需要的字段；请求字符串超过 CTP 字段长度时
直接拒绝，回调先复制原生结构，再由专用线程调用 Python。回调队列溢出时
发送断线通知，使上层停止交易。
