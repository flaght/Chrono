"""
IPyStrategy - 策略基类
与生产系统 bomber/stgWrapper 的 IPyStrategy 接口完全一致。

策略团队继承此类，实现 onBar/onMarketData 等回调。
适配层在底层将其桥接到 Nautilus Strategy。
"""
from bomber_adapter.types import BarData, MarketData, GreeksData, IVData


class IPyStrategy:
    """
    策略基类 - 与生产系统接口完全一致。
    策略开发者继承此类，实现回调函数。
    """

    # 策略配置（子类可以覆盖）
    STRATEGY_NAME = "Base Strategy"
    MULTIPLIER_MAP = {"IC": 200, "IF": 300, "IH": 300, "IM": 200}
    EXTRA_INDEX_CONTRACTS = []  # 额外需要加载的指数
    EXTRA_FUTURES_CONTRACTS = []  # 额外需要加载的期货
    BAR_PERIOD = "M1"            # 策略订阅的K线周期（M1/M5/M15/M30/H1/D1）

    def __init__(self):
        self._stg_name = ""
        self._env = None
        self._data_api = None
        self._subscriptions = {
            "instruments": [],
            "bars": {},       # {inst: [period_str, ...]}
            "iv": [],
            "greeks": [],
        }
        self._target_vol_cb = None
        self._order_cb = None

    # --- 初始化 ---

    def initialize(self, name: str, env):
        self._stg_name = name
        self._env = env

    def get_insts(self) -> list:
        """返回策略需要订阅的所有合约"""
        return []

    def get_strategy_name(self) -> str:
        """返回策略名称"""
        return self.STRATEGY_NAME

    def get_multiplier_map(self) -> dict:
        """返回合约乘数映射"""
        return self.MULTIPLIER_MAP

    def get_extra_contracts(self) -> tuple:
        """
        返回额外需要加载的合约（不在 get_insts() 中，但需要加载数据）

        返回:
            tuple: (extra_index_contracts, extra_futures_contracts)
        """
        return (self.EXTRA_INDEX_CONTRACTS, self.EXTRA_FUTURES_CONTRACTS)

    def set_contracts(self, contract, contracts_df=None):
        """设置当前交易合约（由数据管道在换月时调用）"""
        pass

    # --- 核心回调 ---

    def onBar(self, BarData):
        """
        单根 K 线回调 — 每根 bar 到达时立即推送一次。
        **接口签名与生产系统（bomber/stgWrapper）完全一致**。

        参数:
            BarData: 当前 bar 的 OHLCV 数据 (BarData 对象)

        适用：逐 bar 处理、单合约策略、需要立即响应的场景。

        说明：
            与 onBatchBar **同时生效**：
            - onBar 在每根 bar 到达时立即推送
            - onBatchBar 在该时间戳所有 bar 到齐后再逐个推送
            策略可以只实现其中一个，也可以两个都实现。
        """
        pass

    def onBatchBar(self, inst: str, ts: int, barData, isLast: bool):
        """
        批量 K 线回调 — 同一时间戳同一频率的所有 bar 到齐后逐个推送。
        **接口签名与生产系统（bomber/stgWrapper）完全一致**。

        参数:
            inst:     合约代码（如 "IC2401"）
            ts:       微秒级 unix 时间戳（= 秒 × 1_000_000）
            barData:  当前 bar 的 OHLCV 数据 (BarData 对象)
            isLast:   是否为当前批次的最后一个 bar

        适用：
            - 需要"所有 bar 到齐再处理"的场景
            - 多合约协同策略（通过 isLast 判断是否所有 bar 已推送完毕）
            - 在最后一个 bar 到达时触发批量计算/下单

        与 onBar 的关系：
            对同一根 bar，onBar 和 onBatchBar **都会被调用**：
            1. bar 到达时 → 立即触发 onBar(BarData)
            2. 同 ts 所有 bar 到齐后 → 逐个触发 onBatchBar(inst, ts, barData, isLast)

            策略可以只实现其中一个。如果两个都实现，注意避免重复下单。

        典型用法：
            def onBatchBar(self, inst, ts, barData, isLast):
                if not isLast:
                    return                    # 等最后一个 bar 再处理
                # 此时同一 ts 的所有合约 bar 都已推完
                self.rebalance_portfolio()
        """
        pass

    def onMarketData(self, data: MarketData):
        pass

    # --- 可选回调（期权） ---

    def onIVData(self, data: IVData):
        pass

    def onGreeksData(self, data: GreeksData):
        pass

    # --- 每日回调 ---

    def onDailyOpen(self, date: str):
        """
        每日开盘回调 - 在每天第一根K线到达前调用

        参数:
            date: 交易日期 (YYYY-MM-DD)

        用途:
            加载当日需要刷新的数据，如：
            - 期货公司持仓数据
            - 期权 Greeks
            - 其他每日更新的数据
        """
        pass

    def onDailyClose(self, date: str):
        """
        每日收盘回调 - 在每天最后一根K线到达后调用

        参数:
            date: 交易日期 (YYYY-MM-DD)

        用途:
            执行收盘后的处理，如：
            - 记录当日持仓
            - 计算当日盈亏
            - 清理临时数据
        """
        pass

    # --- 数据订阅 ---

    def subInstrument(self, inst: str):
        if inst not in self._subscriptions["instruments"]:
            self._subscriptions["instruments"].append(inst)

    def subBar(self, inst: str, period: str):
        if inst not in self._subscriptions["bars"]:
            self._subscriptions["bars"][inst] = []
        if period not in self._subscriptions["bars"][inst]:
            self._subscriptions["bars"][inst].append(period)

    def subIV(self, inst: str):
        if inst not in self._subscriptions["iv"]:
            self._subscriptions["iv"].append(inst)

    def subGreeks(self, inst: str):
        if inst not in self._subscriptions["greeks"]:
            self._subscriptions["greeks"].append(inst)

    def subAndGetBars(self, insts, period, start: str, end: str):
        if not isinstance(insts, list):
            insts = [insts]
        for inst in insts:
            self.subBar(inst, str(period))

    def getHistoryBar(self, inst_list: list, period: str, start: str, end: str) -> dict:
        """
        加载历史 K 线数据

        参数:
            inst_list: 合约列表（如 ['IC2609', 'IF2609']）或单个合约（如 'IC2609'）
            period: K 线周期（如 'M1', 'M5', 'D1'）
            start: 开始时间（'YYYY-MM-DD HH:MM:SS'）
            end: 结束时间（'YYYY-MM-DD HH:MM:SS'）

        返回:
            dict: {
                'IC2609': DataFrame,  # 包含 timestamp, open, high, low, close, volume 等列
                'IF2609': DataFrame,
                ...
            }

        使用示例:
            def initialize(self, name, env):
                super().initialize(name, env)

                # 加载历史 K 线数据
                inst_list = ['IC2609', 'IF2609', 'IH2609']
                self.barM1s = self.getHistoryBar(inst_list, 'M1', '2026-01-01 09:00:00', '2026-08-01 09:30:00')
        """
        if self._env is None:
            print("[IPyStrategy] env 未初始化，无法加载历史 K 线")
            return {}

        if not isinstance(inst_list, list):
            inst_list = [inst_list]

        # 调用 data_api 加载历史数据
        return self._env.data_api.load_history_bars(inst_list, period, start, end)

    def getTradingCalendar(self, start_date: str, end_date: str):
        """
        获取交易日历

        参数:
            start_date: 开始日期（'YYYY-MM-DD'）
            end_date: 结束日期（'YYYY-MM-DD'）

        返回:
            DataFrame: 交易日历，包含以下列：
                - date: 日期
                - is_trading_day: 是否为交易日

        使用示例:
            def initialize(self, name, env):
                super().initialize(name, env)

                # 加载交易日历
                self.trading_calendar = self.getTradingCalendar('2020-01-01', '2026-12-31')
                trading_days = self.trading_calendar[self.trading_calendar['is_trading_day']]
        """
        if self._env is None:
            print("[IPyStrategy] env 未初始化，无法加载交易日历")
            import pandas as pd
            return pd.DataFrame(columns=['date', 'is_trading_day'])

        return self._env.data_api.load_calendar(start_date, end_date)

    def getContinuingContracts(self, date: str, product_id: str, algo_id: int = 1) -> dict:
        """
        获取指定日期的主力和次主力合约

        参数:
            date: 交易日期（'YYYY-MM-DD'）
            product_id: 品种 ID（如 'IM', 'IC', 'IF', 'IH'）
            algo_id: 策略 ID（默认 1）

        返回:
            dict: {
                'main_contract': {'code': 'IM2609', 'multiplier_1min': 1.466, 'multiplier_5min': 1.466},
                'sub_contract': {'code': 'IM2612', 'multiplier_1min': 1.441, 'multiplier_5min': 1.441}
            }

        使用示例:
            def onDailyOpen(self, date: str):
                # 获取 IM 品种的主力合约
                info = self.getContinuingContracts(date, 'IM')
                main_code = info['main_contract']['code']  # 如 'IM2609'
                sub_code = info['sub_contract']['code']    # 如 'IM2612'
        """
        if self._env is None:
            print("[IPyStrategy] env 未初始化，无法获取主力合约")
            return {}

        return self._env.data_api.load_continuing_multiplier(date, product_id, algo_id)

    # --- 下单 ---

    def sendTargetVol(self, code: str, vol: float, timestamp: int, isLast: bool):
        if self._target_vol_cb:
            self._target_vol_cb(code, vol, timestamp, isLast)

    def sendOrder(self, code: str, direction: int, volume: float, timestamp: int):
        if self._order_cb:
            self._order_cb(code, direction, volume, timestamp)

    def sendTargetPos(self, code, reserved, target_pos, timestamp, isLast):
        self.sendTargetVol(code, target_pos, timestamp, isLast)

    def get_position(self, code: str) -> float:
        """查询当前持仓（从回测引擎实时获取）"""
        if self._env is not None:
            return self._env.get_position(code)
        return 0.0

    def get_positions(self) -> dict:
        """
        获取所有当前持仓信息（从回测引擎实时获取）

        返回:
            dict: {合约代码: 持仓数量}，正数表示多头，负数表示空头

        示例:
            positions = self.get_positions()
            # 返回: {'IC2403': 1.0, 'IF2403': -2.0}
            for code, qty in positions.items():
                print(f"{code}: {qty}")
        """
        if self._env is not None:
            return self._env.get_positions()
        return {}

    # --- 合约信息查询 ---

    @staticmethod
    def get_product(inst: str) -> str:
        """获取合约品种代码（如 IC2401 → "IC"，MO2401-C-5000 → "MO"）"""
        from bomber_adapter.instrument_info import get_product
        return get_product(inst)

    @staticmethod
    def is_option(inst: str) -> bool:
        """判断是否为期权合约"""
        from bomber_adapter.instrument_info import is_option
        return is_option(inst)

    @staticmethod
    def is_future(inst: str) -> bool:
        """判断是否为期货合约（非期权）"""
        from bomber_adapter.instrument_info import is_future
        return is_future(inst)

    @staticmethod
    def is_index(inst: str) -> bool:
        """判断是否为指数合约（SH000905 中证500 等）"""
        from bomber_adapter.instrument_info import is_index
        return is_index(inst)

    @staticmethod
    def get_contract_type(inst: str) -> str:
        """返回合约类型：FUTURE / OPTION / INDEX / UNKNOWN"""
        from bomber_adapter.instrument_info import get_contract_type
        return get_contract_type(inst)

    @staticmethod
    def get_multiplier(inst: str):
        """获取合约乘数，未知品种返回 None"""
        from bomber_adapter.instrument_info import get_multiplier
        return get_multiplier(inst)

    @staticmethod
    def get_underlying(inst: str):
        """获取期权对应的标的品种（MO → IM）；期货返回 None"""
        from bomber_adapter.instrument_info import get_underlying
        return get_underlying(inst)

    @staticmethod
    def get_underlying_contract(inst: str):
        """获取期权对应的标的期货合约（MO2401-C-5000 → IM2401）；期货返回 None"""
        from bomber_adapter.instrument_info import get_underlying_contract
        return get_underlying_contract(inst)

    @staticmethod
    def get_option_info(inst: str):
        """解析期权合约完整信息（product/expiry/option_type/strike/underlying/multiplier），期货返回 None"""
        from bomber_adapter.instrument_info import get_option_info
        return get_option_info(inst)

    def exit(self):
        pass

    def _reset_state(self):
        """重置策略内部状态（由桥接器在 on_reset 时调用）"""
        pass
