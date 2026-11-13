import time,pdb
from dotenv import load_dotenv

load_dotenv()

from kernel.monitor import Monitor


def main():
    pdb.set_trace()
    monitor = Monitor(name='ctp')
    monitor.load_accounts_from_toml("/workspace/worker/pj/Chrono/entangle/config/account.toml")
    monitor.start(delay_between_accounts=1.5)
    print("\n[运行中] monitor 插件已启动，正在统一维护资金、持仓、委托、成交和持仓明细。按 Ctrl+C 可停止运行。\n")
    pdb.set_trace()
    try:
        while True:
            #time.sleep(10)
            monitor.print_dashboard()
            monitor.monitor_plugin.print_dashboard()
            monitor.monitor_plugin.save_account_fund_snapshots(
                db_name='nurex_daily',
                table_name='AccountFundSnapshots',
            )
    except KeyboardInterrupt:
        print("\n收到退出信号，正在停止监控...")
        monitor.close()
        print("监控已安全退出。")


if __name__ == "__main__":
    main()
