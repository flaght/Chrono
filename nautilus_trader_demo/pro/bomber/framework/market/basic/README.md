# market/basic 编译说明

`base.py` 加载 `custom_bar` 和 `fast_factory` 两个 Cython 扩展。首次部署，以及修改 `.pyx`、`.pxd` 或更换 Bomber 内核、Python、操作系统、架构后，需要重新编译。编译与运行使用同一个 Python 环境。

环境需要已编译并可导入的 `bomber` 内核及其 `.pxd`，以及 Cython、setuptools、wheel、NumPy 和 C 编译器。两个扩展直接 `cimport bomber.*`。

从 pro 根目录编译：

```bash
export PYTHONPATH=.
python bomber/framework/market/basic/setup.py build_ext --inplace
```

也可以在本目录执行 `python setup.py build_ext --inplace`。脚本自动定位包含 `bomber` 的包根目录，从实际加载的 Bomber 包获取 Cython 接口搜索路径，不依赖固定的 `code/bomber` 目录。

扩展名称固定为 `bomber.framework.market.basic.custom_bar` 和 `bomber.framework.market.basic.fast_factory`，`--inplace` 将产物写入本目录。修改后可加 `--force` 重编译。

```bash
python bomber/framework/market/basic/setup.py build_ext --inplace --force
python - <<'PY'
import bomber.framework.market.basic.custom_bar as custom_bar
import bomber.framework.market.basic.fast_factory as fast_factory
print(custom_bar.__file__)
print(fast_factory.__file__)
PY
```

路径应指向当前 framework 的 `.so` 或 `.pyd`。发布脚本同步 `.pyx`、`.pxd` 和构建脚本，排除本目录生成的 C 文件、构建缓存与扩展二进制；目标环境需要重新构建。若复制到 Bomber 主库中，其 `build.py` 已递归收集 `bomber/**/*.pyx`，也可以随主库统一构建。
