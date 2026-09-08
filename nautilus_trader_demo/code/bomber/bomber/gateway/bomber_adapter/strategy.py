"""
IPyStrategy - 策略基类
与生产系统 bomber/stgWrapper 的 IPyStrategy 接口完全一致。

策略团队继承此类，实现 onBar/onMarketData 等回调。
适配层在底层将其桥接到 Nautilus Strategy。
"""
from .types import BarData, MarketData, GreeksData, IVData


class IPyStrategy:
    """
    策略基类 - 与生产系统接口完全一致。
    策略开发者继承此类，实现回调函数。
    """

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
        return []

    def set_contracts(self, contract, contracts_df=None):
        """设置当前交易合约（由数据管道在换月时调用）"""
        pass

    # --- 核心回调 ---

    def onBar(self, bar_data: BarData):
        pass

    def onBatchBar(self, inst: str, ts: int, bar_data: BarData, isLast: bool):
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

    def exit(self):
        pass

    def _reset_state(self):
        """重置策略内部状态（由桥接器在 on_reset 时调用）"""
        pass
