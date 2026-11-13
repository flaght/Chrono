import time,os,pdb

from dotenv import load_dotenv

load_dotenv()

from vnpy.event import EventEngine, Event
from vnpy.trader.engine import MainEngine
from vnpy.trader.object import AccountData
from vnpy.trader.event import EVENT_ACCOUNT, EVENT_LOG

from vnpy_ctp import CtpGateway

class AccountEquityManager:
    """账户权益管理器"""
    def __init__(self, main_engine: MainEngine, event_engine: EventEngine):
        self.main_engine = main_engine
        self.event_engine = event_engine
        
        # 注册事件监听
        self.register_event()
    def register_event(self) -> None:
        """注册资金回报事件"""
        self.event_engine.register(EVENT_ACCOUNT, self.process_account_event)
        self.event_engine.register(EVENT_LOG, self.process_log_event)
    def process_account_event(self, event: Event) -> None:
        """
        回调函数：当接口推送账户资金更新时触发
        """
        account: AccountData = event.data
        print(f"\n[资金更新通知] 收到账户权益推送:")
        print(f"  - 账户: {account.vt_accountid}")
        print(f"  - 总权益 (Balance): {account.balance:,.2f}")
        print(f"  - 可用资金 (Available): {account.available:,.2f}")
        print(f"  - 冻结资金 (Frozen): {account.frozen:,.2f}")
    def process_log_event(self, event: Event) -> None:
        """日志回调"""
        log = event.data
        # print(f"[{log.time}] {log.msg}")
    def get_equity_by_id(self, vt_accountid: str) -> AccountData:
        """
        主动获取指定账户的资金数据
        :param vt_accountid: 全局账户ID，例如 "CTP.012345"
        """
        account = self.main_engine.get_account(vt_accountid)
        return account
    def print_all_equities(self) -> None:
        """主动遍历打印所有账户资金"""
        accounts = self.main_engine.get_all_accounts()
        if not accounts:
            print("当前暂无已缓存的账户数据（可能尚未连接或未收到推送）。")
            return
        print("\n===== 当前所有账户权益信息 =====")
        for acc in accounts:
            print(
                f"账户: {acc.vt_accountid} | "
                f"总资产: {acc.balance:,.2f} | "
                f"可用: {acc.available:,.2f} | "
                f"冻结: {acc.frozen:,.2f}"
            )
        print("================================")
        
def main():
    # 1. 创建事件引擎与主引擎
    event_engine = EventEngine()
    main_engine = MainEngine(event_engine)
    # 2. 实例化账户管理器
    manager = AccountEquityManager(main_engine, event_engine)
    # 3. 添加对应的 Gateway（以 CTP 为例）
    if CtpGateway:
        main_engine.add_gateway(CtpGateway)
        # 4. 配置连接参数（根据实际信息填写）
        pdb.set_trace()
        ctp_setting = {
            "用户名": os.environ['CTP_ACCOUNT_ID'],
            "密码": os.environ['CTP_PASSWORD'],
            "经纪商代码": os.environ['CTP_BROKER_ID'],
            "交易服务器": os.environ['CTP_TD_ADDRESS'],
            "行情服务器": os.environ['CTP_MD_ADDRESS'],
            "产品名称": os.environ['CTP_APP_ID'],
            "授权编码":os.environ['CTP_AUTH_CODE']
        }
        # 5. 连接接口（登录成功后，接口底层会自动发送查资金请求并触发 EVENT_ACCOUNT）
        main_engine.connect(ctp_setting, "CTP")
    # 等待接口登录并接收首次推送
    time.sleep(10)
    # 6. 主动查询方式
    manager.print_all_equities()
    # 如果知道具体 vt_accountid，可以直接查询：
    # acc = manager.get_equity_by_id("CTP.your_investor_id")
    # if acc:
    #     print("查询到总权益:", acc.balance)
    # 保持主线程运行以持续接收推送
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        main_engine.close()
if __name__ == "__main__":
    main()