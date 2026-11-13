import os
import time
from dotenv import load_dotenv

load_dotenv()

from kernel.monitor import Monitor


def resolve_account_config():
    """按显式环境变量、项目配置目录的顺序寻找账户配置。"""
    explicit = os.environ.get("ENTANGLE_ACCOUNT_CONFIG")
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
    configured_root = os.environ.get("ENTL_CONFIG", project_root)
    candidates = [
        explicit,
        os.path.join(configured_root, "config", "accounts.toml"),
        os.path.join(configured_root, "config", "account.toml"),
        os.path.join(project_root, "config", "accounts.toml"),
        os.path.join(project_root, "config", "account.toml"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "未找到CTP账户配置。请设置 ENTANGLE_ACCOUNT_CONFIG，"
        "或创建 config/accounts.toml（兼容 config/account.toml）。"
    )


def main():
    monitor = Monitor(name='ctp')
    monitor.load_accounts_from_toml(resolve_account_config())
    monitor.start(delay_between_accounts=1.5)
    print("\n[运行中] monitor 插件已启动，正在统一维护资金、持仓、委托、成交和持仓明细。按 Ctrl+C 可停止运行。\n")
    try:
        while True:
            time.sleep(10)
            monitor.print_dashboard()
            monitor.monitor_plugin.print_dashboard()
            monitor.monitor_plugin.save_monitor_snapshots()
    except KeyboardInterrupt:
        print("\n收到退出信号，正在停止监控...")
        monitor.close()
        print("监控已安全退出。")


if __name__ == "__main__":
    main()
