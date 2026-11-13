# -*- coding: utf-8 -*-
"""
多账号权益与持仓监控核心模块
支持从 TOML 配置中加载多账号凭据及 ProductID 产品标识，
通过事件驱动模型实时监听资金 (EVENT_ACCOUNT) 与持仓 (EVENT_POSITION) 回报，
并将时间统一校准为北京时间 (UTC+8)，支持转为 DataFrame 写入 DolphinDB (AccountFundSnapshots 表)。
基于 ultron.tradingday.advanceDateByCalendar 精准判定交易日与节假日开休市，
在夜盘时段（自动推演至下一交易日）、周五夜盘（推演至下周一）及节假日期间保持完全合规。
"""

import os, importlib, time, datetime, pdb, toml
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd
from vnpy.event import Event, EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.object import AccountData, PositionData, LogData
from vnpy.trader.event import EVENT_ACCOUNT, EVENT_POSITION, EVENT_LOG
from kdutil.dolphindb1 import DolphinDBAdapter
from toolix.plugins import load_monitor_plugin

# 中国北京时区 (UTC+8)
CHINA_TZ = datetime.timezone(datetime.timedelta(hours=8))


def get_china_now() -> datetime.datetime:
    """
    获取当前北京时间 (UTC+8)，返回 naive datetime 以避免在跨平台或入库时发生二次时区偏移
    """
    return datetime.datetime.now(CHINA_TZ).replace(tzinfo=None)


def _advance_trading_date(anchor_date: datetime.date, offset: str) -> datetime.date:
    """
    调用 ultron.tradingday.advanceDateByCalendar 计算交易日偏移 (考虑法定节假日与周末)。
    若未安装 ultron 则自动回退到常规周末计算。
    :param offset: 例如 "+1b" (下一个交易日), "-1b" (上一个交易日), "0b" (当前或最近交易日)
    """
    try:
        from ultron.tradingday import advanceDateByCalendar
        res = advanceDateByCalendar("china.sse", anchor_date, offset)
        if isinstance(res, (datetime.datetime, pd.Timestamp)):
            return res.date()
        return res
    except ImportError:
        # 回退基础周末日历算法
        days = 1 if "+1" in offset else (-1 if "-1" in offset else 0)
        target = anchor_date + datetime.timedelta(days=days)
        if "+1" in offset:
            while target.weekday() >= 5:  # 周六或周日跳过
                target += datetime.timedelta(days=1)
        elif "-1" in offset:
            while target.weekday() >= 5:
                target -= datetime.timedelta(days=1)
        return target


def _is_market_trade_date(d: datetime.date) -> bool:
    """判断某一自然日是否为中国期货/证券市场的法定交易日"""
    try:
        from ultron.tradingday import advanceDateByCalendar
        # 若以 d 计算 -1b 再 +1b 等于 d，说明 d 本身就是交易日
        prev_b = advanceDateByCalendar("china.sse", d, "-1b")
        if isinstance(prev_b, (datetime.datetime, pd.Timestamp)):
            prev_b = prev_b.date()
        next_b = advanceDateByCalendar("china.sse", prev_b, "+1b")
        if isinstance(next_b, (datetime.datetime, pd.Timestamp)):
            next_b = next_b.date()
        return next_b == d
    except ImportError:
        return d.weekday() < 5


def get_china_trading_day(dt: Optional[datetime.datetime] = None) -> str:
    """
    计算当前时刻对应的国内期货市场标准交易日 (格式: YYYY-MM-DD)
    借助 ultron.tradingday.advanceDateByCalendar 交易日历支持，自动精准处理节假日与周末：
    1. 晚间夜盘 (16:30 - 24:00)：通过 advanceDateByCalendar("+1b") 获取下一个交易日（自动跳过周末及长假）
    2. 凌晨夜盘延续 (00:00 - 03:00)：取前一天 (dt - 1天) 的下一个交易日 ("+1b")
    3. 白天日盘时段 (03:00 - 16:30)：如果当天是交易日则为当天，非交易日则取下一个交易日
    """
    if dt is None:
        dt = get_china_now()

    current_date = dt.date()
    current_time = dt.time()

    # 1. 凌晨 00:00 - 03:00 (前一日夜盘延续)
    if current_time < datetime.time(3, 0, 0):
        yesterday = current_date - datetime.timedelta(days=1)
        trading_date = _advance_trading_date(yesterday, "+1b")
    # 2. 晚间 16:30 - 24:00 (夜盘时段)
    elif current_time >= datetime.time(16, 30, 0):
        trading_date = _advance_trading_date(current_date, "+1b")
    # 3. 白天 03:00 - 16:30 (日盘时段)
    else:
        if _is_market_trade_date(current_date):
            trading_date = current_date
        else:
            # 若在周末/节假日白天运行，归入节后第一个交易日
            trading_date = _advance_trading_date(current_date, "+1b")

    return trading_date.strftime("%Y-%m-%d")


def is_trading_time(dt: Optional[datetime.datetime] = None) -> bool:
    """
    判断当前时间是否处于国内期货交易时段 (北京时间 UTC+8)
    结合节假日日历精确判定：
    1. 凌晨夜盘延续 (00:00:00 - 02:30:30)：前一天必须是交易日
    2. 白天日盘时段 (08:59 - 10:15, 10:29 - 11:30, 13:29 - 15:15)：当天必须是交易日
    3. 晚间夜盘时段 (20:59:00 - 23:59:59)：当天必须是交易日且节前最后一个交易日晚间无夜盘
    """
    if dt is None:
        dt = get_china_now()

    current_date = dt.date()
    current_time = dt.time()

    # 1. 凌晨夜盘延续时段 (00:00:00 - 02:30:30)
    if datetime.time(0, 0, 0) <= current_time <= datetime.time(2, 30, 30):
        yesterday = current_date - datetime.timedelta(days=1)
        return _is_market_trade_date(yesterday)

    # 2. 白天日盘时段 (必须为法定交易日)
    if not _is_market_trade_date(current_date):
        return False

    # 早盘第一节 (08:59:00 - 10:15:30)
    if datetime.time(8, 59, 0) <= current_time <= datetime.time(10, 15, 30):
        return True

    # 早盘第二节 (10:29:00 - 11:30:30)
    if datetime.time(10, 29, 0) <= current_time <= datetime.time(11, 30, 30):
        return True

    # 下午盘 (13:29:00 - 15:15:30)
    if datetime.time(13, 29, 0) <= current_time <= datetime.time(15, 15, 30):
        return True

    # 3. 晚间夜盘阶段 (20:59:00 - 23:59:59)
    if datetime.time(20, 59, 0) <= current_time <= datetime.time(23, 59, 59):
        # 国内期货规则：长假前最后一个交易日当晚不设夜盘 (普通周五有夜盘，因周六凌晨收盘)
        next_b = _advance_trading_date(current_date, "+1b")
        # 若下一个交易日跨度大于3天（长假如国庆/春节/五一），节前当晚休市
        if (next_b - current_date).days > 3:
            return False
        return True

    return False


def _patch_ctp_gateway_fields(gateway_module):
    """
    针对 vnpy_ctp 网关注入增强补丁：
    vnpy 原生 AccountData 默认丢弃了 CTP 柜台返回的 CloseProfit, Commission,
    Deposit, Withdraw, PositionProfit, FrozenMargin 等丰富字段。
    本补丁将 CTP 原始字段完整保留到 account 实例上，供数据看板和入库直接使用。
    """
    try:
        # 递归寻找 CtpTdApi 类
        td_api_cls = getattr(gateway_module, "CtpTdApi", None)
        if not td_api_cls:
            for attr in dir(gateway_module):
                obj = getattr(gateway_module, attr)
                if isinstance(obj, type) and "TdApi" in attr:
                    td_api_cls = obj
                    break

        if td_api_cls:
            # 1. 拦截登录回报，捕获 CTP 柜台系统当前真实的权威 TradingDay
            if hasattr(td_api_cls, "onRspUserLogin"):
                orig_on_login = td_api_cls.onRspUserLogin

                def patched_onRspUserLogin(self, data: dict, error: dict, reqid: int, last: bool):
                    if data and "TradingDay" in data and data["TradingDay"]:
                        self.gateway.trading_day = data["TradingDay"]
                    orig_on_login(self, data, error, reqid, last)

                td_api_cls.onRspUserLogin = patched_onRspUserLogin

            # 2. 拦截资金查询回报，提取完整资金明细
            if hasattr(td_api_cls, "onRspQryTradingAccount"):
                def patched_onRspQryTradingAccount(self, data: dict, error: dict, reqid: int, last: bool):
                    if not data:
                        return
                    account = AccountData(
                        accountid=data["AccountID"],
                        balance=data["Balance"],
                        frozen=data.get("FrozenMargin", 0.0) + data.get("FrozenCash", 0.0) + data.get("FrozenCommission", 0.0),
                        available=data["Available"],
                        gateway_name=self.gateway_name
                    )
                    # 保留完整的 CTP 柜台原始数据字典
                    account.extra = data
                    account.close_profit = float(data.get("CloseProfit", 0.0))
                    account.position_profit = float(data.get("PositionProfit", 0.0))
                    account.commission = float(data.get("Commission", 0.0))
                    account.margin = float(data.get("CurrMargin", 0.0))
                    account.deposit = float(data.get("Deposit", 0.0))
                    account.withdraw = float(data.get("Withdraw", 0.0))
                    account.frozen_margin = float(data.get("FrozenMargin", 0.0))
                    # 使用网关登录时的权威交易日（而非历史结算单日期）
                    account.trading_day = getattr(self.gateway, "trading_day", "")
                    self.gateway.on_account(account)

                td_api_cls.onRspQryTradingAccount = patched_onRspQryTradingAccount
    except Exception as e:
        print(f"[提示] CTP 网关字段增强补丁注入异常: {e}")


class Monitor(object):

    def __init__(self, name='ctp'):
        self.name = name
        self.ddb_adapter = DolphinDBAdapter(uri=os.environ.get("DDB_URL", os.environ.get("DDB_URI", "")))
        self.event_engine = EventEngine()
        self.event_engine.register(EVENT_LOG, self.process_log_event)
        self.event_engine.register(EVENT_ACCOUNT, self.process_account_event)
        self.event_engine.register(EVENT_POSITION, self.process_position_event)

        self.main_engine = MainEngine(self.event_engine)
        module_name = "vnpy_{0}".format(self.name)

        try:
            self.module_class = importlib.import_module(module_name)
            self.getway_name = "{0}Gateway".format(self.name.capitalize())
            self.getway_module = self.module_class.__getattribute__(
                self.getway_name)
            # 自动注入 CTP 字段补丁
            _patch_ctp_gateway_fields(self.module_class)
        except ImportError as e:
            raise ImportError(str(e))

        # 缓存：vt_accountid -> AccountData
        self.accounts: Dict[str, AccountData] = {}
        # 缓存：gateway_name -> {vt_positionid: PositionData}
        self.positions: Dict[str, Dict[str, PositionData]] = defaultdict(dict)
        # 账号配置注册表：gateway_name -> dict(config=..., product_id=..., account_id=...)
        self.account_configs: Dict[str, dict] = {}
        # 记录各网关由 CTP 返回的最新真实交易日 (TradingDay)
        self.trading_days: Dict[str, str] = {}
        # 记录上次成功写入 DolphinDB 的 UpdateTime，防止同一 5 分钟 Bar 重复写入
        self.last_saved_update_time: Optional[datetime.datetime] = None
        # 上次主动查询 CTP 资金的时间戳，用于控制查询频率避免触发 CTP 流控
        self._last_query_time: float = 0.0
        # 上次收到 EVENT_ACCOUNT 事件的时间戳 (time.time())，用于检测 CTP 是否已断开
        self._last_account_event_time: float = 0.0

        # 统一监控插件共享本 MainEngine、EventEngine 和账户配置。
        # 插件本身不会创建或连接任何额外 CTP Gateway。
        self.monitor_plugin = load_monitor_plugin(
            main_engine=self.main_engine,
            event_engine=self.event_engine,
            account_configs=self.account_configs,
            gateway_module=self.module_class,
            storage_adapter=self.ddb_adapter,
            market_time_checker=is_trading_time,
            trading_day_resolver=get_china_trading_day,
        )

    def load_accounts_from_toml(self, toml_path: str = "config/accounts.toml") -> List[str]:
        """
        从 TOML 配置文件批量加载并注册账号
        支持 [[accounts]] 列表格式，提取 ProductID / product_id 用于后续数据表映射
        """
        if not os.path.exists(toml_path):
            raise FileNotFoundError(f"未找到 TOML 配置文件: {toml_path}")

        data = toml.load(toml_path)
        accounts = data.get("accounts", [])
        if isinstance(accounts, dict):
            acc_list = []
            for gw_name, acc in accounts.items():
                if "gateway_name" not in acc:
                    acc["gateway_name"] = gw_name
                acc_list.append(acc)
            accounts = acc_list

        registered = []
        for acc in accounts[0:1]:
            product_id = str(acc.get("product_id") or acc.get("ProductID") or "")
            account_id = str(acc.get("account_id") or acc.get("AccountID") or "")
            password = str(acc.get("password") or acc.get("Password") or "")
            broker_id = str(acc.get("broker_id") or acc.get("BrokerID") or "")
            td_address = str(acc.get("td_address") or acc.get("TdAddress") or "")
            md_address = str(acc.get("md_address") or acc.get("MdAddress") or "")
            app_id = str(acc.get("app_id") or acc.get("AppID") or "")
            auth_code = str(acc.get("auth_code") or acc.get("AuthCode") or "")
            gw_name = acc.get("gateway_name") or acc.get("GatewayName")

            gw_name = self.add_account(
                account_id=account_id,
                password=password,
                broker_id=broker_id,
                td_address=td_address,
                md_address=md_address,
                app_id=app_id,
                auth_code=auth_code,
                product_id=product_id,
                gateway_name=gw_name
            )
            registered.append(gw_name)
        print(f"[{get_china_now().strftime('%H:%M:%S')}] 成功从 TOML 文件 ({toml_path}) 加载并注册了 {len(registered)} 个账号: {registered}")
        return registered
            
    def add_account(self, account_id: str, password: str, broker_id: str,
                    td_address: str, md_address: str, app_id: str,
                    auth_code: str, product_id: str = "",
                    gateway_name: Optional[str] = None) -> str:
        """
        注册单个账户
        :param product_id: 产品/基金/投资组合标识（映射到数据表的 ProductID）
        :param gateway_name: 自定义网关唯一名称，如未指定则自动按序号递增 (如 CTP_01, CTP_02)
        """
        if not gateway_name:
            gateway_name = f"{self.name.upper()}_{len(self.account_configs) + 1:02d}"

        # 添加网关实例到 MainEngine（不同 gateway_name 对应独立实例）
        self.main_engine.add_gateway(self.getway_module, gateway_name)

        config = {
            "用户名": account_id,
            "密码": password,
            "经纪商代码": broker_id,
            "交易服务器": td_address,
            "行情服务器": md_address,
            "产品名称": app_id,
            "授权编码": auth_code
        }
        self.account_configs[gateway_name] = {
            "config": config,
            "product_id": product_id,
            "account_id": account_id,
            "broker_id": broker_id
        }
        return gateway_name

    def connect_account(self, gateway_name: str) -> None:
        """连接指定的账号网关"""
        if gateway_name not in self.account_configs:
            raise ValueError(f"未找到网关 {gateway_name} 的配置信息！")
        acc_info = self.account_configs[gateway_name]
        config = acc_info["config"]
        product_id = acc_info.get("product_id", "")
        print(
            f"[{get_china_now().strftime('%H:%M:%S')}] 正在发起连接: {gateway_name} (ProductID: {product_id}, 账号: {config['用户名']})..."
        )
        self.main_engine.connect(config, gateway_name)
        # connect() 会注册 vnpy_ctp 自带的资金/持仓轮询；立即交由统一插件接管，
        # 避免多账户依次连接期间就发生未受控查询。
        self.monitor_plugin.attach_gateway(gateway_name)

    def query_all_accounts(self, min_interval: float = 60.0, query_delay: float = 1.0) -> None:
        """
        主动向所有 CTP 网关发起资金与持仓查询请求 (reqQryTradingAccount / reqQryInvestorPosition)
        CTP 柜台不会自动推送资金变动，必须由客户端主动发起查询才能获取最新权益数据。
        若不定期调用此方法，self.accounts 缓存将始终停留在登录时的静态快照。
        :param min_interval: 两次查询之间的最小间隔秒数 (避免触发 CTP 每秒 1 次查询的流控限制)
        :param query_delay: 多网关间查询间隔秒数 (CTP 流控规定：同一 TdApi 连续请求间隔 >= 1 秒)
        """
        now_ts = time.time()
        if now_ts - self._last_query_time < min_interval:
            return
        self._last_query_time = now_ts
        self.monitor_plugin.request_tables(("funds", "positions"))

    def request_full_snapshot(self, gateway_name=None, wait=False,
                              timeout=30.0, ready_timeout=120.0):
        """通过 monitor 插件查询五张表，不创建额外 CTP 连接。"""
        return self.monitor_plugin.request_full_snapshot(
            gateway_name=gateway_name,
            wait=wait,
            timeout=timeout,
            ready_timeout=ready_timeout,
        )

    def get_monitor_table(self, table_name: str) -> pd.DataFrame:
        """获取 funds/positions/orders/trades/position_details 表。"""
        return self.monitor_plugin.get_dataframe(table_name)

    def process_log_event(self, event: Event) -> None:
        """日志回调"""
        log: LogData = event.data
        print(f"[{get_china_now().strftime('%Y-%m-%d %H:%M:%S.%f')}] [{log.gateway_name}] {log.msg}")
        
    def process_account_event(self, event: Event) -> None:
        """资金变动回报回调"""
        acc: AccountData = event.data
        # 记录最后一次收到 CTP 资金回报的时间（用于检测连接是否存活）
        self._last_account_event_time = time.time()
        # 绑定 ProductID 属性，方便后续直接取用或写入 AccountFundSnapshots 表
        cfg_info = self.account_configs.get(acc.gateway_name, {})
        acc.product_id = cfg_info.get("product_id", "")
        self.accounts[acc.vt_accountid] = acc

        # 实时计算风险度 = (总权益 - 可用资金) / 总权益 * 100%
        risk_ratio = 0.0
        if acc.balance > 0:
            used_margin = max(0.0, acc.balance - acc.available)
            risk_ratio = (used_margin / acc.balance) * 100.0

        if risk_ratio >= 80.0:
            print(
                f"\n⚠️  [高风险预警] ProductID: {acc.product_id} | 账号: {acc.vt_accountid} 保证金占用率达 {risk_ratio:.2f}% (总权益: {acc.balance:,.2f}, 可用: {acc.available:,.2f})"
            )
            
    def process_position_event(self, event: Event) -> None:
        """持仓变动回报回调"""
        pos: PositionData = event.data
        gw = pos.gateway_name or "DEFAULT"

        # 绑定 ProductID
        cfg_info = self.account_configs.get(gw, {})
        pos.product_id = cfg_info.get("product_id", "")

        # 若持仓量归零（已全部平仓），清理缓存
        if pos.volume == 0:
            if pos.vt_positionid in self.positions[gw]:
                del self.positions[gw][pos.vt_positionid]
            return

        # 更新实时持仓
        self.positions[gw][pos.vt_positionid] = pos

    def get_summary(self) -> dict:
        """获取全账户聚合统计"""
        total_balance = 0.0
        total_available = 0.0
        total_frozen = 0.0
        total_pnl = 0.0
        total_positions_count = 0

        for acc in self.accounts.values():
            total_balance += acc.balance
            total_available += acc.available
            total_frozen += acc.frozen

        for gw, pos_dict in self.positions.items():
            total_positions_count += len(pos_dict)
            for pos in pos_dict.values():
                total_pnl += getattr(pos, "pnl", 0.0)

        total_margin = max(0.0, total_balance - total_available)
        overall_risk = (total_margin / total_balance * 100.0) if total_balance > 0 else 0.0

        return {
            "account_count": len(self.accounts),
            "total_balance": total_balance,
            "total_available": total_available,
            "total_frozen": total_frozen,
            "total_margin": total_margin,
            "overall_risk": overall_risk,
            "total_positions_count": total_positions_count,
            "total_pnl": total_pnl,
            "update_time": get_china_now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def print_dashboard(self) -> None:
        """
        在控制台打印当前格式化的多账号权益与持仓看板
        """
        summary = self.get_summary()
        sep_thick = "=" * 98
        sep_thin = "-" * 98

        trading_status = "【盘中交易中】" if is_trading_time() else "【已收盘/休市中】"
        trading_day_disp = get_china_trading_day()

        print(f"\n{sep_thick}")
        print(f"       多账号实时监控看板 ({summary['update_time']}) | 交易日: {trading_day_disp} {trading_status}")
        print(f"{sep_thick}")

        # 1. 账号权益板块
        print("【账户资金与风险度监控】")
        print(f"{'产品ID(ProductID)':<16} | {'全局账户ID':<18} | {'总权益(Balance)':>14} | {'可用资金(Avail)':>14} | {'冻结(Frozen)':>12} | {'风险度':>8}")
        print(sep_thin)

        if not self.accounts:
            print("  (暂无已连接账户或未收到资金推送)")
        else:
            for acc in self.accounts.values():
                product_id = getattr(acc, "product_id", "") or "-"
                risk = ((acc.balance - acc.available) / acc.balance * 100.0) if acc.balance > 0 else 0.0
                print(
                    f"{product_id:<16} | "
                    f"{acc.vt_accountid:<18} | "
                    f"{acc.balance:>14,.2f} | "
                    f"{acc.available:>14,.2f} | "
                    f"{acc.frozen:>12,.2f} | "
                    f"{risk:>7.1f}%"
                )

        # 2. 持仓板块
        print(f"\n【当前有效持仓明细】")
        print(f"{'产品ID':<12} | {'网关/账户':<10} | {'合约代码':<12} | {'方向':<4} | {'总持仓':>6} | {'昨仓':>6} | {'持仓均价':>10} | {'浮动盈亏':>12}")
        print(sep_thin)

        has_position = False
        for gw, pos_map in self.positions.items():
            for pos in pos_map.values():
                has_position = True
                product_id = getattr(pos, "product_id", "") or "-"
                dir_str = pos.direction.value if hasattr(pos.direction, "value") else str(pos.direction)
                pnl = getattr(pos, "pnl", 0.0)
                print(
                    f"{product_id:<12} | "
                    f"{gw:<10} | "
                    f"{pos.vt_symbol:<12} | "
                    f"{dir_str:<4} | "
                    f"{pos.volume:>6} | "
                    f"{pos.yd_volume:>6} | "
                    f"{pos.price:>10.2f} | "
                    f"{pnl:>12,.2f}"
                )

        if not has_position:
            print("  (当前所有账户无持仓)")

        # 3. 汇总板块
        print(sep_thin)
        print(f"【全账户汇总看板】")
        print(
            f"监控账户数: {summary['account_count']:<3} | "
            f"汇总总权益: {summary['total_balance']:>14,.2f} | "
            f"汇总可用: {summary['total_available']:>14,.2f} | "
            f"全盘风险度: {summary['overall_risk']:>5.1f}% | "
            f"总持仓浮盈亏: {summary['total_pnl']:>12,.2f}"
        )
        print(f"{sep_thick}\n")

    def to_dataframe(self, interval_minutes: int = 5) -> pd.DataFrame:
        """
        将当前所有账户的最新资金权益快照转换为 pandas.DataFrame
        字段顺序与命名严格契合 AccountFundSnapshots 数据库表结构:
        [UpdateTime, TradingDay, ProductID, AccountID, BrokerID, Balance, DynamicEquity,
         Margin, Available, CloseProfit, PositionProfit, Commission, FrozenMargin, Deposit, Withdraw]
        时间统一为北京时间 (UTC+8)，UpdateTime 向下对齐到 5 分钟 Bar，TradingDay 严格遵循中国期货交易日历（支持节假日）。
        """
        records = []
        now = get_china_now()
        
        # 将分钟向下对齐到 5 的倍数（如 10:16:18 -> 10:15:00），秒和微秒归零
        if interval_minutes > 1:
            aligned_minute = (now.minute // interval_minutes) * interval_minutes
            update_time = now.replace(minute=aligned_minute, second=0, microsecond=0)
        else:
            update_time = now

        # 核心交易日计算：借助交易日历权威计算当前时刻对应的中国期货交易日 (考虑周末及节假日)
        computed_trading_day = get_china_trading_day(now)

        for acc in self.accounts.values():
            cfg_info = self.account_configs.get(acc.gateway_name, {})
            product_id = cfg_info.get("product_id", "") or getattr(
                acc, "product_id", "")
            broker_id = cfg_info.get("broker_id", "")

            # 交易日处理：如果网关在登录时获取到了合法的最新柜台交易日且有效，优先采用；否则采用日历推算的交易日
            gateway_login_day = getattr(acc, "trading_day", "") or self.trading_days.get(acc.gateway_name, "")
            if gateway_login_day and len(gateway_login_day) == 8 and gateway_login_day.isdigit():
                formatted_gw_day = f"{gateway_login_day[:4]}-{gateway_login_day[4:6]}-{gateway_login_day[6:]}"
                if formatted_gw_day >= computed_trading_day:
                    trading_day_str = formatted_gw_day
                else:
                    trading_day_str = computed_trading_day
            else:
                trading_day_str = computed_trading_day

            # 读取 CTP 柜台原始数据 extra 字典
            extra = getattr(acc, "extra", {}) or {}

            balance = float(getattr(acc, "balance", 0.0))
            available = float(getattr(acc, "available", 0.0))
            
            # 冻结保证金
            frozen_margin = float(
                getattr(acc, "frozen_margin", 0.0) or
                extra.get("FrozenMargin", 0.0) or
                getattr(acc, "frozen", 0.0)
            )
            
            # 当前持仓保证金
            margin = float(
                getattr(acc, "margin", 0.0) or
                extra.get("CurrMargin", 0.0) or
                max(0.0, balance - available)
            )
            
            # 平仓盈亏
            close_profit = float(
                getattr(acc, "close_profit", 0.0) or
                extra.get("CloseProfit", 0.0)
            )
            
            # 手续费
            commission = float(
                getattr(acc, "commission", 0.0) or
                extra.get("Commission", 0.0)
            )
            
            # 入金 & 出金
            deposit = float(
                getattr(acc, "deposit", 0.0) or
                extra.get("Deposit", 0.0)
            )
            withdraw = float(
                getattr(acc, "withdraw", 0.0) or
                extra.get("Withdraw", 0.0)
            )

            # 持仓浮动盈亏：优先从 CTP 资金结构体获取，若为 0 则从当前持仓列表中累加求和
            raw_pos_profit = float(
                getattr(acc, "position_profit", 0.0) or
                extra.get("PositionProfit", 0.0)
            )
            if raw_pos_profit != 0.0:
                position_profit = raw_pos_profit
            else:
                pos_map = self.positions.get(acc.gateway_name, {})
                position_profit = float(sum(getattr(p, "pnl", 0.0) for p in pos_map.values()))

            # 动态权益
            dynamic_equity = float(
                getattr(acc, "dynamic_equity", 0.0) or
                (balance + position_profit if position_profit != 0.0 else balance)
            )

            records.append({
                "UpdateTime": update_time,
                "TradingDay": trading_day_str,
                "ProductID": product_id,
                "AccountID": str(acc.accountid),
                "BrokerID": str(broker_id),
                "Balance": balance,
                "DynamicEquity": dynamic_equity,
                "Margin": margin,
                "Available": available,
                "CloseProfit": close_profit,
                "PositionProfit": position_profit,
                "Commission": commission,
                "FrozenMargin": frozen_margin,
                "Deposit": deposit,
                "Withdraw": withdraw
            })

        cols = [
            "UpdateTime", "TradingDay", "ProductID", "AccountID", "BrokerID",
            "Balance", "DynamicEquity", "Margin", "Available", "CloseProfit",
            "PositionProfit", "Commission", "FrozenMargin", "Deposit",
            "Withdraw"
        ]
        if not records:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(records)[cols]

    def is_ctp_alive(self, stale_timeout: float = 180.0) -> bool:
        """
        判断 CTP 柜台连接是否仍然存活
        如果在最近 stale_timeout 秒内没有收到任何 EVENT_ACCOUNT 回报，
        则认为 CTP 已经断开（夜盘收盘、网络中断等）。
        :param stale_timeout: 判定为断连的超时秒数 (默认 180 秒 = 3 分钟)
        """
        if self._last_account_event_time <= 0:
            return False  # 从未收到过资金回报
        return (time.time() - self._last_account_event_time) < stale_timeout

    def save_to_dolphindb(self,
                          db_name: str = "account_monitor",
                          table_name: str = "AccountFundSnapshots",
                          force: bool = False) -> bool:
        """兼容入口：实际写入由 monitor 插件统一完成。"""
        return self.monitor_plugin.save_account_fund_snapshots(
            db_name=db_name,
            table_name=table_name,
            force=force,
        )

    def close(self):
        """安全关闭所有网关与引擎"""
        print("正在关闭监控中心与底层网关...")
        self.monitor_plugin.stop()
        self.main_engine.close()
        
    def start(self, delay_between_accounts: float = 1.5) -> None:
        """
        批量连接所有注册的账号网关
        :param delay_between_accounts: 多账号连接间隔秒数，避免并发冲击柜台
        """
        total = len(self.account_configs)
        print(f"开始批量连接 {total} 个账号...")
        for i, gw_name in enumerate(self.account_configs.keys(), 1):
            self.connect_account(gw_name)
            if i < total and delay_between_accounts > 0:
                time.sleep(delay_between_accounts)
        self.monitor_plugin.start()
