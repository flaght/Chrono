"""旧指数探针命令兼容入口；实现统一由配置驱动只读入口承担。"""
import sys
from .read_only import main as read_only_main


def main(argv=None):
    return read_only_main(["--dataset", "index.minute", "--key", "000852",
        *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
