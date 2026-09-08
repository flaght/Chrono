"""
BomberBridge - 核心桥接器
继承 Nautilus Strategy，内部持有用户的 IPyStrategy，
将 Nautilus 事件转换为 bomber 回调格式。
"""
import hashlib
import math
from decimal import Decimal
from datetime import datetime, timezone

from bomber.trading.strategy import Strategy
from bomber.model.data import Bar, BarType, QuoteTick, TradeTick
from bomber.model.identifiers import InstrumentId, Symbol, Venue
from bomber.model.objects import Price, Quantity
from bomber.model.enums import OrderSide
from bomber.common.enums import LogColor

from .types import BarData, MarketData, GreeksData, IVData
from .enums import bar_time_unit_t, PERIOD_MAP, OrderMsg_dir_t, OrderMsg_offset_t


class BomberBridge(Strategy):
    """Nautilus Strategy 子类，桥接用户的 IPyStrategy。"""

    def __init__(self, user_strategy, data_api=None, data_dir=None):
        super().__init__()
        self._user_stg = user_strategy
        self._data_api = data_api
        self._data_dir = data_dir or "./data"

        self._last_market = {}
        self._pending_targets = {}
        self._pending_positions = {}  # 本批次内已提交但 cache 尚未反映的持仓
        self._last_sent_targets = {}  # 已成功发送的目标持仓（跨 bar 持久），防次日重复
        self._current_day = None

        self._user_stg._target_vol_cb = self._on_send_target_vol
        self._user_stg._order_cb = self._on_send_order

    # ================================================================
    # Nautilus 生命周期
    # ================================================================

    def on_start(self):
        from .env import BacktestEnv
        env = BacktestEnv(data_dir=self._data_dir)
        env.set_bridge(self)
        self._user_stg.initialize(self._user_stg._stg_name, env)

        if self._data_api:
            self._user_stg._data_api = self._data_api

        subs = self._user_stg._subscriptions

        for inst in subs.get("instruments", []):
            inst_id = self._resolve_inst_id(inst)
            self.subscribe_quote_ticks(inst_id)
            self.subscribe_trade_ticks(inst_id)

        for inst, periods in subs.get("bars", {}).items():
            inst_id = self._resolve_inst_id(inst)
            for period in periods:
                bar_type = self._resolve_bar_type(inst, period)
                self.subscribe_bars(bar_type)

        self.log.info(f"BomberBridge started: {self._user_stg._stg_name}",
                       color=LogColor.YELLOW)

    def on_stop(self):
        if self._current_day:
            self._user_stg.onDailyClose(self._current_day)

    def on_reset(self):
        self._last_market.clear()
        self._pending_targets.clear()
        self._pending_positions.clear()
        self._last_sent_targets.clear()
        self._current_day = None

    # ================================================================
    # Nautilus 事件 → 用户回调
    # ================================================================

    def on_bar(self, bar: Bar):
        inst_str = bar.bar_type.instrument_id.symbol.value

        # 日期检测用 UTC（避免本地时区偏移）
        bar_date = datetime.fromtimestamp(int(bar.ts_event) / 1e9, tz=timezone.utc).strftime('%Y-%m-%d')

        if self._current_day is None:
            self._current_day = bar_date
            self._user_stg.onDailyOpen(bar_date)
        elif bar_date != self._current_day:
            self._user_stg.onDailyClose(self._current_day)
            self._current_day = bar_date
            self._last_sent_targets.clear()  # 新交易日重置
            self._user_stg.onDailyOpen(bar_date)

        bar_spec = bar.bar_type.spec
        time_unit = self._bar_spec_to_time_unit(bar_spec.aggregation, bar_spec.step)

        bar_type_str = str(bar.bar_type)
        bar_code_hash = hashlib.sha256(bar_type_str.encode()).digest()
        bar_code = int.from_bytes(bar_code_hash[:4], 'big')

        bar_data = BarData(
            inst=inst_str, ts=int(bar.ts_event),
            timeUnit=time_unit, barCode=bar_code,
            openPrice=float(bar.open), highPrice=float(bar.high),
            lowPrice=float(bar.low), closePrice=float(bar.close),
            volume=float(bar.volume), turnOver=0.0,
        )
        bar_data._nautilus_bar = bar

        self._user_stg.onBar(bar_data)

        # onBatchBar: isLast 始终为 True，因为策略自行通过 sendTargetVol 的 isLast 控制批量提交
        ts_us = int(bar.ts_event) // 1000
        self._user_stg.onBatchBar(inst_str, ts_us, bar_data, True)

    def on_quote_tick(self, tick: QuoteTick):
        inst_str = tick.instrument_id.symbol.value
        prev = self._last_market.get(inst_str)
        mid_price = (float(tick.bid_price) + float(tick.ask_price)) / 2.0

        md = MarketData(
            inst=inst_str, ts=int(tick.ts_event),
            exchangeTSMicro=int(tick.ts_event) // 1000, Level=1,
            lastPrice=mid_price, volume=0.0, turnover=0.0, openinterest=0.0,
            bidPrice=[float(tick.bid_price)], bidVolume=[float(tick.bid_size)],
            askPrice=[float(tick.ask_price)], askVolume=[float(tick.ask_size)],
            upLimitPrice=prev.upLimitPrice if prev else 0.0,
            lowLimitPrice=prev.lowLimitPrice if prev else 0.0,
            preSettlePrice=prev.preSettlePrice if prev else 0.0,
            openPrice=prev.openPrice if prev else 0.0,
        )
        self._last_market[inst_str] = md
        self._user_stg.onMarketData(md)

    def on_trade_tick(self, tick: TradeTick):
        inst_str = tick.instrument_id.symbol.value
        prev = self._last_market.get(inst_str)

        md = MarketData(
            inst=inst_str, ts=int(tick.ts_event),
            exchangeTSMicro=int(tick.ts_event) // 1000, Level=1,
            lastPrice=float(tick.price), volume=float(tick.size),
            turnover=float(tick.price) * float(tick.size), openinterest=0.0,
            bidPrice=prev.bidPrice if prev else [float(tick.price)],
            bidVolume=prev.bidVolume if prev else [float(tick.size)],
            askPrice=prev.askPrice if prev else [float(tick.price)],
            askVolume=prev.askVolume if prev else [float(tick.size)],
            upLimitPrice=prev.upLimitPrice if prev else 0.0,
            lowLimitPrice=prev.lowLimitPrice if prev else 0.0,
            preSettlePrice=prev.preSettlePrice if prev else 0.0,
            openPrice=prev.openPrice if prev else 0.0,
        )
        self._last_market[inst_str] = md
        self._user_stg.onMarketData(md)

    def on_data(self, data):
        if hasattr(data, "_bomber_type"):
            if data._bomber_type == "greeks":
                self._user_stg.onGreeksData(GreeksData(
                    InstStr=data._inst, delta=data._delta, gamma=data._gamma,
                    theta=data._theta, vega=data._vega, rho=data._rho,
                    impliedVolatility=data._iv, timestamp=data._ts,
                ))
            elif data._bomber_type == "iv":
                self._user_stg.onIVData(IVData(
                    InstStr=data._inst, vol=data._vol,
                    price=data._price, timestamp=data._ts,
                ))

    # ================================================================
    # 用户下单 → Nautilus 订单
    # ================================================================

    def _on_send_target_vol(self, code, vol, timestamp, isLast):
        self._pending_targets[code] = (vol, timestamp)
        if isLast:
            self._flush_targets()

    def _flush_targets(self):
        if not self._pending_targets:
            return

        for code, (target_vol, ts) in list(self._pending_targets.items()):
            inst_id = self._resolve_inst_id(code)

            # 相同目标已在本次 DailyOpen 中发送过，跳过
            if code in self._last_sent_targets and self._last_sent_targets[code] == target_vol:
                continue

            # 实际持仓 = cache + 本批次累积增量
            current_qty = self.get_position(code)
            pending = self._pending_positions.get(code, 0.0)
            effective_qty = current_qty + pending

            if not math.isfinite(target_vol):
                self.log.warning(f"Invalid target_vol for {code}: {target_vol}")
                continue

            delta = target_vol - effective_qty
            if abs(delta) < 1e-8:
                self._last_sent_targets[code] = target_vol
                continue

            instrument = self.cache.instrument(inst_id)
            if instrument is None:
                self.log.warning(f"Instrument not found: {code} (inst_id={inst_id})")
                continue

            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            qty = instrument.make_qty(abs(delta))

            self.log.debug(f"Submitting order: {code} {side} qty={qty} delta={delta}")

            order = self.order_factory.market(
                instrument_id=inst_id, order_side=side, quantity=qty,
            )
            self.submit_order(order)

            # 记录本批次增量 & 已发送目标
            self._pending_positions[code] = pending + delta
            self._last_sent_targets[code] = target_vol

        self._pending_targets.clear()
        self._pending_positions.clear()

    def _on_send_order(self, code, direction, volume, timestamp):
        inst_id = self._resolve_inst_id(code)
        instrument = self.cache.instrument(inst_id)
        if instrument is None:
            self.log.warning(f"Instrument not found: {code}")
            return
        side = OrderSide.BUY if direction == OrderMsg_dir_t.Buy else OrderSide.SELL
        qty = instrument.make_qty(abs(volume))
        order = self.order_factory.market(
            instrument_id=inst_id, order_side=side, quantity=qty,
        )
        self.submit_order(order)

    # ================================================================
    # 辅助方法
    # ================================================================

    def get_position(self, code: str) -> float:
        inst_id = self._resolve_inst_id(code)
        try:
            positions = self.cache.positions(venue=inst_id.venue)
            for pos in positions:
                if pos.instrument_id == inst_id:
                    return float(pos.net_qty.as_double())
        except Exception:
            pass
        return 0.0

    def _resolve_inst_id(self, inst: str) -> InstrumentId:
        if "." in inst:
            return InstrumentId.from_str(inst)
        return InstrumentId(Symbol(inst), Venue("CFFEX"))

    def _resolve_bar_type(self, inst: str, period: str) -> BarType:
        inst_id = self._resolve_inst_id(inst)
        agg_name, step = PERIOD_MAP.get(period, ("MINUTE", 1))
        bar_type_str = f"{inst_id}-{step}-{agg_name}-LAST-EXTERNAL"
        return BarType.from_str(bar_type_str)

    def _bar_spec_to_time_unit(self, agg, step: int) -> int:
        try:
            agg_name = agg.name
        except AttributeError:
            from bomber.model.data import BarAggregation
            try:
                agg_name = BarAggregation(agg).name
            except (ValueError, TypeError):
                return bar_time_unit_t.M1
        mapping = {
            ("SECOND", 1): bar_time_unit_t.S1, ("SECOND", 5): bar_time_unit_t.S5,
            ("SECOND", 15): bar_time_unit_t.S15, ("MINUTE", 1): bar_time_unit_t.M1,
            ("MINUTE", 5): bar_time_unit_t.M5, ("MINUTE", 15): bar_time_unit_t.M15,
            ("HOUR", 1): bar_time_unit_t.H1, ("DAY", 1): bar_time_unit_t.D1,
            ("WEEK", 1): bar_time_unit_t.W1,
        }
        return mapping.get((agg_name, step), bar_time_unit_t.M1)
