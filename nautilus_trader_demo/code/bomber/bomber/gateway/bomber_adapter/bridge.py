"""
BomberBridge - 核心桥接器（支持 delay_bars 执行延迟）
继承 Nautilus Strategy，内部持有用户的 IPyStrategy，
将 Nautilus 事件转换为 bomber 回调格式。

delay_bars:
  0: 当前 bar close 撮合（默认）
  1: 下一根 bar close 撮合
  N: 延迟 N 根 bar 后撮合
"""
import hashlib
import math
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from bomber.trading.strategy import Strategy
from bomber.model.data import Bar, BarType, QuoteTick, TradeTick
from bomber.model.identifiers import InstrumentId, Symbol, Venue
from bomber.model.objects import Price, Quantity
from bomber.model.enums import OrderSide
from bomber.common.enums import LogColor

from bomber_adapter.types import BarData, MarketData, GreeksData, IVData
from bomber_adapter.enums import bar_time_unit_t, PERIOD_MAP, OrderMsg_dir_t, OrderMsg_offset_t
from bomber_adapter.instrument_info import is_option


class BomberBridge(Strategy):
    """Nautilus Strategy 子类，桥接用户的 IPyStrategy。"""

    def __init__(self, user_strategy, data_api=None, data_dir=None,
                 execution_delay_bars: int = 0, venue: str = "CFFEX",
                 config: dict = None, option_expiry_dates: dict = None,
                 futures_expiry_dates: dict = None,
                 bar_extra_fields: dict = None):
        super().__init__()
        if execution_delay_bars < 0:
            raise ValueError("execution_delay_bars must be non-negative")
        self._user_stg = user_strategy
        self._data_api = data_api
        env_data_dir = os.environ.get("BOMBER_DATA_DIR") or os.environ.get("BT_DATA_DIR")
        self._data_dir = data_dir or env_data_dir or "./data"
        self._execution_delay_bars = execution_delay_bars
        self._venue = Venue(venue)
        self._tz = ZoneInfo("Asia/Shanghai")
        self._config = config or {}  # 保存配置，供 BacktestEnv 使用
        # 期权到期日：{symbol: "YYYY-MM-DD"}，由 engine._option_expiry_dates 传入
        self._option_expiry_dates = option_expiry_dates or {}
        # 期货到期日：{symbol: "YYYY-MM-DD"}，由 engine._futures_expiry_dates 传入
        self._futures_expiry_dates = futures_expiry_dates or {}
        # Bar 额外字段 sidecar 字典：{(instrument_id, timestamp_ns): {field: value}}
        self._bar_extra_fields = bar_extra_fields or {}

        self._last_market = {}
        self._pending_targets = {}
        self._last_flush_ts = 0
        self._current_day = None
        self._last_bar_ts = {}
        self._last_bar_close = {}
        self._bar_sequences = {}
        self._delayed_targets = {}
        self._delayed_orders = []
        # 同一批次提交中的累积持仓调整（防止连续调用 sendTargetVol 时计算错误）
        self._pending_position_adjustments = {}
        self._active_bar_ts = None
        self._expected_bar_types = set()
        self._batch_ts = None
        self._batch_bars = defaultdict(list)
        # 性能优化：bar_code 缓存，避免每个 bar 重复计算 SHA256
        self._bar_code_cache = {}

        # 持仓记录：记录每个时刻的持仓状态
        self._position_history = []
        self._fill_occurred = False  # 标记是否有订单成交
        self._last_fill_ts = None  # 上一次成交的时间戳

        self._user_stg._target_vol_cb = self._on_send_target_vol
        self._user_stg._order_cb = self._on_send_order

    # ================================================================
    # Nautilus 生命周期
    # ================================================================

    def on_start(self):
        from bomber_adapter.env import BacktestEnv
        # 过滤掉已显式传递的参数
        env_params = {k: v for k, v in self._config.items() if k not in ('data_dir',)}
        env = BacktestEnv(data_dir=self._data_dir, **env_params)
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
            if self.cache.instrument(inst_id) is None:
                raise ValueError(f"Subscribed instrument was not loaded: {inst_id}")
            for period in periods:
                bar_type = self._resolve_bar_type(inst, period)
                self._expected_bar_types.add(str(bar_type))
                self.subscribe_bars(bar_type)

        self.log.info(f"BomberBridge started: {self._user_stg._stg_name}",
                       color=LogColor.YELLOW)

    def on_stop(self):
        # Submit all pending targets before stopping
        # This ensures that targets sent at the last bar (e.g., 15:00) are executed
        self._flush_bar_batch(submit_targets=True)
        if self._pending_targets:
            self.log.warning(
                f"Discarding {len(self._pending_targets)} targets created at end of backtest",
            )
            self._pending_targets.clear()
        if self._delayed_targets or self._delayed_orders:
            self.log.warning(
                "Backtest ended with delayed orders: "
                f"targets={len(self._delayed_targets)}, orders={len(self._delayed_orders)}"
            )
        if self._current_day:
            # 在回测结束时，检查并平掉到期合约
            self._close_expiring_contracts(self._current_day)
            self._user_stg.onDailyClose(self._current_day)

    def on_reset(self):
        self._last_market.clear()
        self._pending_targets.clear()
        self._bar_sequences.clear()
        self._last_bar_ts.clear()
        self._last_bar_close.clear()
        self._delayed_targets.clear()
        self._delayed_orders.clear()
        self._expected_bar_types.clear()
        self._batch_bars.clear()
        self._batch_ts = None
        self._current_day = None
        self._last_flush_ts = 0

    # ================================================================
    # Nautilus 事件 → 用户回调
    # ================================================================

    def on_bar(self, bar: Bar):
        # 如果有订单成交，记录实际持仓快照
        if self._fill_occurred:
            self._record_actual_position_snapshot(self._last_fill_ts)
            self._fill_occurred = False
            self._last_fill_ts = None

        inst_id = bar.bar_type.instrument_id
        key = str(inst_id)
        ts = int(bar.ts_event)

        # 关键修改：在处理新交易日之前，先检查是否需要记录日终持仓快照
        # 此时 _last_bar_ts 还是前一天的最后一个 bar 时间戳
        bar_date = datetime.fromtimestamp(ts / 1e9, tz=self._tz).strftime('%Y-%m-%d')
        if self._current_day is not None and bar_date != self._current_day:
            # 新交易日开始，记录前一天的日终持仓快照
            if self._last_bar_ts:
                last_ts = max(self._last_bar_ts.values()) if self._last_bar_ts else None
                self._record_actual_position_snapshot(last_ts)

        if self._batch_ts is not None and ts != self._batch_ts:
            self._flush_bar_batch()
        sequence = self._bar_sequences.get(key, 0) + 1
        self._bar_sequences[key] = sequence
        self._last_bar_ts[key] = ts
        self._last_bar_close[key] = float(bar.close)  # 记录每个合约的最后一个bar收盘价

        self._submit_due_orders(key, sequence, ts)
        self._roll_trading_day(ts)

        bar_spec = bar.bar_type.spec
        bar_type_str = str(bar.bar_type)
        # 性能优化：使用缓存避免每个 bar 重复计算 SHA256
        bar_code = self._bar_code_cache.get(bar_type_str)
        if bar_code is None:
            bar_code_hash = hashlib.sha256(bar_type_str.encode()).digest()
            bar_code = int.from_bytes(bar_code_hash[:4], 'big')
            self._bar_code_cache[bar_type_str] = bar_code

        # 从 sidecar 字典获取额外字段
        extra_key = (inst_id.symbol.value, ts)
        extra = self._bar_extra_fields.get(extra_key, {})

        bar_data = BarData(
            inst=inst_id.symbol.value, ts=ts,
            timeUnit=self._bar_spec_to_time_unit(
                bar_spec.aggregation,
                bar_spec.step,
            ),
            barCode=bar_code,
            openPrice=float(bar.open), highPrice=float(bar.high),
            lowPrice=float(bar.low), closePrice=float(bar.close),
            volume=float(bar.volume),
            turnOver=extra.get("turnover", 0.0),
            # 期货累计字段
            turnoverAccumulate=extra.get("turnoverAccumulate", 0.0),
            volumeAccumulate=extra.get("volumeAccumulate", 0.0),
            openInterestAccumulate=extra.get("openInterestAccumulate", 0.0),
            # 指数分段价格字段
            sectionalLowPrice=extra.get("sectionalLowPrice", 0.0),
            sectionalHighPrice=extra.get("sectionalHighPrice", 0.0),
            sectionalOpenPrice=extra.get("sectionalOpenPrice", 0.0),
        )
        bar_data._nautilus_bar = bar

        self._active_bar_ts = ts
        try:
            self._user_stg.onBar(bar_data)
        finally:
            self._active_bar_ts = None

        # 按时间戳收集 batch：同一时间戳的所有 bar 一起推送
        # 性能优化：使用 defaultdict，无需检查 key 是否存在
        self._batch_bars[ts].append((inst_id.symbol.value, bar_data))
        self._batch_ts = ts

    def _flush_bar_batch(self, submit_targets=True):
        if not self._batch_bars:
            return

        # 按时间戳顺序处理所有 batch
        # 性能优化：单 ts 场景（大多数情况）跳过排序
        if len(self._batch_bars) == 1:
            timestamps = list(self._batch_bars.keys())
        else:
            timestamps = sorted(self._batch_bars.keys())

        for ts in timestamps:
            items = self._batch_bars[ts]
            self._active_bar_ts = ts
            try:
                # 方式一：逐个推送（兼容原接口）
                for index, (inst, bar_data) in enumerate(items):
                    self._user_stg.onBatchBar(
                        inst,
                        int(ts) // 1000,
                        bar_data,
                        index == len(items) - 1,
                    )
                # 方式二：批量推送（新接口）
                if hasattr(self._user_stg, 'onBatchBars'):
                    batch_bar_dict = {inst: bar_data for inst, bar_data in items}
                    self._user_stg.onBatchBars(batch_bar_dict)
            finally:
                self._active_bar_ts = None

        self._batch_bars.clear()
        self._batch_ts = None
        if submit_targets:
            self._flush_targets(implicit_close=True)

    def _roll_trading_day(self, ts):
        bar_date = datetime.fromtimestamp(ts / 1e9, tz=self._tz).strftime('%Y-%m-%d')
        if self._current_day is None:
            self._current_day = bar_date
            self._user_stg.onDailyOpen(bar_date)
        elif bar_date != self._current_day:
            # 在调用 onDailyClose 前，检查并平掉到期合约
            self._close_expiring_contracts(self._current_day)

            # 日终持仓快照已在 on_bar 中记录（使用前一天的最后一个 bar 时间戳）

            self._user_stg.onDailyClose(self._current_day)
            self._current_day = bar_date
            self._user_stg.onDailyOpen(bar_date)

    def _close_expiring_contracts(self, date_str: str):
        """
        检查并平掉到期合约的持仓

        在 onDailyClose 前调用，自动平掉当天到期的期货和期权持仓。
        """
        try:
            # 获取所有当前持仓
            positions = list(self.cache.positions())
            if not positions:
                return

            expiring_positions = []

            for position in positions:
                inst_id = position.instrument_id
                code = str(inst_id.symbol)

                # 跳过空持仓
                qty = float(position.quantity)
                if abs(qty) < 1e-8:
                    continue

                # 检查是否到期
                is_expiring = False

                # 期权：使用 _option_expiry_dates
                if is_option(code):
                    exp_day = self._option_expiry_dates.get(code)
                    if exp_day and exp_day == date_str:
                        is_expiring = True

                # 期货：使用 _futures_expiry_dates（预加载）
                else:
                    futures_exp_day = self._futures_expiry_dates.get(code)
                    if futures_exp_day and futures_exp_day == date_str:
                        is_expiring = True

                if is_expiring:
                    expiring_positions.append((code, qty))

            # 平掉到期合约
            if expiring_positions:
                print(f"[Bridge] {date_str} 发现 {len(expiring_positions)} 个到期合约需要平仓")
                for code, qty in expiring_positions:
                    # 发送目标持仓为 0，触发平仓
                    self._submit_target(code, 0.0)
                    print(f"[Bridge]   平仓到期合约: {code} (原持仓: {qty:+.0f})")

                # 关键修改：到期平仓后记录持仓快照（此时持仓可能已变为0）
                # 使用当前时间戳
                self._record_actual_position_snapshot()

        except Exception as e:
            print(f"[Bridge] 检查到期合约失败: {e}")
            import traceback
            traceback.print_exc()

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

    def on_order_filled(self, event):
        """记录订单成交事件，并标记需要记录持仓快照"""
        super().on_order_filled(event)
        # 标记有订单成交
        self._fill_occurred = True
        # 记录成交时间戳
        self._last_fill_ts = int(event.ts_event) if hasattr(event, 'ts_event') else None

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
        # 跨 slot 保护
        if self._pending_targets and self._last_flush_ts != timestamp:
            self._flush_targets(implicit_close=False)
        self._pending_targets[code] = (vol, timestamp)
        self._last_flush_ts = timestamp
        if isLast:
            self._flush_targets(implicit_close=True)

    def _due_sequence(self, inst_id):
        key = str(inst_id)
        sequence = self._bar_sequences.get(key, 0)
        cross_contract_offset = (
            1
            if self._active_bar_ts is not None
            and self._last_bar_ts.get(key) != self._active_bar_ts
            else 0
        )
        return sequence + self._execution_delay_bars + cross_contract_offset

    def _flush_targets(self, implicit_close: bool = False):
        pending = self._pending_targets
        self._pending_targets = {}
        # 清空同一批次的累积调整，开始新的计算周期
        self._pending_position_adjustments = {}

        if not pending:
            return

        # 标准化 pending 中的 code（统一格式用于后续比较）
        normalized_pending = {}
        for code, (target_vol, ts) in pending.items():
            norm_code = code if '.' in code else code + '.CFFEX'
            normalized_pending[norm_code] = (target_vol, ts)

        # 1. 处理显式目标
        for code, (target_vol, ts) in pending.items():
            if not math.isfinite(target_vol):
                self.log.warning(f"Invalid target_vol for {code}: {target_vol}")
                continue

            inst_id = self._resolve_inst_id(code)
            if self._execution_delay_bars:
                self._delayed_targets[str(inst_id)] = (
                    code,
                    self._due_sequence(inst_id),
                    target_vol,
                    ts,
                    self._active_bar_ts,
                )
            else:
                if not self._submit_target(code, target_vol):
                    self.log.error(
                        f"Target order was not submitted: {code} -> {target_vol}",
                    )

        # 2. 隐式平仓：处理实际持仓中有但目标列表中没有的合约
        # 期货：未到期 → 平仓；已到期 → 跳过（由 _close_expiring_contracts 处理）
        # 期权：未到期 → 保持持仓（target=0 可能表示"保持"）
        #       已到期 → 平仓（过期合约不应继续持有）
        if implicit_close:
            # 性能优化：缓存当前日期（整个 flush 过程中不变）
            current_date = None
            if self._active_bar_ts is not None:
                current_date = datetime.fromtimestamp(
                    self._active_bar_ts / 1e9, tz=self._tz
                ).strftime('%Y-%m-%d')

            for pos in self.cache.positions():
                code = str(pos.instrument_id)
                if '.' not in code:
                    code = code + '.CFFEX'

                # 检查合约是否在目标列表中
                if code not in normalized_pending:
                    inst_clean = code.replace('.CFFEX', '')
                    is_opt = is_option(inst_clean)

                    if is_opt:
                        # 期权：检查是否到期
                        if not self._is_option_expired(inst_clean, self._active_bar_ts):
                            # 未到期期权：保持持仓
                            continue
                        # 已到期期权：继续执行平仓
                    else:
                        # 期货：检查是否到期（使用预计算的 current_date）
                        futures_exp_day = self._futures_expiry_dates.get(inst_clean)
                        if futures_exp_day and current_date and current_date >= futures_exp_day:
                            # 已到期期货：跳过（由 _close_expiring_contracts 处理）
                            continue
                        # 未到期期货：继续执行平仓

                    # 需要平仓
                    current_qty = float(pos.quantity)
                    if not pos.is_long:
                        current_qty = -current_qty

                    if abs(current_qty) > 1e-8:
                        # 需要平仓
                        if self._execution_delay_bars:
                            inst_id = self._resolve_inst_id(code)
                            self._delayed_targets[str(inst_id)] = (
                                code,
                                self._due_sequence(inst_id),
                                0.0,  # 目标为0，即平仓
                                0,
                                self._active_bar_ts,
                            )
                        else:
                            if not self._submit_target(code, 0.0):
                                self.log.warning(f"Failed to close position: {code}")

        # 不再在这里记录快照，改为在订单成交后记录实际持仓
        # 见 on_order_filled 和 on_bar 中的逻辑

    def _on_send_order(self, code, direction, volume, timestamp):
        if not math.isfinite(volume) or volume <= 0:
            self.log.warning(f"Invalid order volume for {code}: {volume}")
            return
        inst_id = self._resolve_inst_id(code)
        if self._execution_delay_bars:
            self._delayed_orders.append((
                code,
                str(inst_id),
                self._due_sequence(inst_id),
                direction,
                volume,
                timestamp,
                self._active_bar_ts,
            ))
        elif not self._submit_direct_order(code, direction, volume):
            self.log.error(f"Direct order was not submitted: {code}")

    def _submit_due_orders(self, key: str, sequence: int, current_bar_ts: int):
        """提交已到期的延迟订单（当前 bar OHLC 已处理完）"""
        delayed = self._delayed_targets.get(key)
        if delayed:
            code, due_seq, target_vol, _, signal_ts = delayed
            if (due_seq <= sequence
                    and (signal_ts is None or current_bar_ts > signal_ts)):
                if self._submit_target(code, target_vol):
                    del self._delayed_targets[key]

        remaining = []
        for delayed in self._delayed_orders:
            code, order_key, due_seq, direction, volume, _, signal_ts = delayed
            if (order_key == key and due_seq <= sequence
                    and (signal_ts is None or current_bar_ts > signal_ts)):
                if not self._submit_direct_order(code, direction, volume):
                    remaining.append(delayed)
            else:
                remaining.append(delayed)
        self._delayed_orders = remaining

    def _submit_target(self, code, target_vol):
        """提交目标持仓订单（delta 计算）"""
        # 防御：已到期期权禁止再成交（无论开仓/调仓/平仓）
        # 到期日当天（== expiry）仍允许最后交易，到期后（> expiry）一律拒绝。
        if self._active_bar_ts is not None:
            clean = code.split('.')[0]
            if is_option(clean):
                exp_day = (self._option_expiry_dates or {}).get(clean)
                if exp_day:
                    current = datetime.fromtimestamp(
                        self._active_bar_ts / 1e9, tz=self._tz,
                    ).strftime('%Y-%m-%d')
                    if current > exp_day:
                        self.log.warning(
                            f"Rejected target for expired option {code}: "
                            f"expiry {exp_day} < bar date {current}, target {target_vol}",
                        )
                        return False
        inst_id = self._resolve_inst_id(code)
        current_qty = self.get_effective_position(code)
        delta = target_vol - current_qty
        if abs(delta) < 1e-8:
            return True
        instrument = self.cache.instrument(inst_id)
        if instrument is None:
            self.log.warning(f"Instrument not found: {code}")
            return False
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        qty = instrument.make_qty(abs(delta))
        order = self.order_factory.market(instrument_id=inst_id, order_side=side, quantity=qty)
        self.submit_order(order)
        # 记录本次提交的持仓调整，供同一批次后续调用使用
        # 例如：先 BUY 10，再 SELL 15，第二次调用时需要看到第一次的 BUY 10
        norm_code = code if '.' in code else code + '.CFFEX'
        self._pending_position_adjustments[norm_code] = (
            self._pending_position_adjustments.get(norm_code, 0.0) + delta
        )
        return True

    def _submit_direct_order(self, code, direction, volume):
        """提交直接订单（已有方向/数量）"""
        inst_id = self._resolve_inst_id(code)
        instrument = self.cache.instrument(inst_id)
        if instrument is None:
            self.log.warning(f"Instrument not found: {code}")
            return False
        side = OrderSide.BUY if direction == OrderMsg_dir_t.Buy else OrderSide.SELL
        qty = instrument.make_qty(abs(volume))
        order = self.order_factory.market(instrument_id=inst_id, order_side=side, quantity=qty)
        self.submit_order(order)
        return True

    def _record_actual_position_snapshot(self, timestamp_ns: int = None):
        """记录实际持仓快照（在订单成交后调用）"""
        try:
            actual_positions = {}
            actual_prices = {}  # 记录每个合约的价格

            for pos in self.cache.positions():
                code = str(pos.instrument_id)
                # 过滤已过期的期权（expiration_ns=∞ 导致 Nautilus 不会自动清除）
                if self._is_option_expired(code, timestamp_ns):
                    continue
                qty = float(pos.quantity)
                if abs(qty) > 1e-8:
                    if '.' not in code:
                        code = code + '.CFFEX'
                    actual_positions[code] = qty if pos.is_long else -qty

                    # 获取该合约的最后一个bar收盘价
                    price = self._last_bar_close.get(code) or self._last_bar_close.get(code.replace('.CFFEX', ''))
                    if price and price > 0:
                        actual_prices[code] = price

            # 使用成交时间戳，如果没有则使用当前 bar 时间戳
            ts = timestamp_ns if timestamp_ns else self._active_bar_ts
            # 关键修改：即使actual_positions为空也记录快照（表示持仓变为0）
            if ts:
                self._position_history.append({
                    'timestamp': ts,
                    'positions': actual_positions.copy(),
                    'prices': actual_prices.copy(),  # 新增价格字段
                })
        except Exception as e:
            self.log.warning(f"Failed to record actual position snapshot: {e}")

    # ================================================================
    # 辅助方法
    # ================================================================

    def get_position(self, code: str) -> float:
        """从 Nautilus cache 获取已成交净仓位。"""
        inst_id = self._resolve_inst_id(code)
        total = 0.0
        for position in self.cache.positions(instrument_id=inst_id):
            qty = float(position.quantity)
            total += qty if position.is_long else -qty
        return total

    def get_positions(self) -> dict:
        """
        获取所有当前持仓信息。

        返回:
            dict: {合约代码: 持仓数量}，正数表示多头，负数表示空头
        """
        positions = {}
        for position in self.cache.positions():
            code = str(position.instrument_id)
            # 移除交易所后缀，保持与策略层一致的格式
            if '.' in code:
                code = code.split('.')[0]
            qty = float(position.quantity)
            # 如果是空头持仓，转为负数
            if not position.is_long:
                qty = -qty
            # 只记录非零持仓
            if abs(qty) > 1e-8:
                positions[code] = qty
        return positions

    def get_effective_position(self, code: str) -> float:
        """已成交仓位加未完成订单剩余量，防止重复下单。"""
        inst_id = self._resolve_inst_id(code)
        total = self.get_position(code)
        for order in self.cache.orders_open(instrument_id=inst_id):
            leaves = float(order.leaves_qty)
            total += leaves if order.side == OrderSide.BUY else -leaves
        # 加上同一批次中已提交但尚未进入 cache 的订单调整
        norm_code = code if '.' in code else code + '.CFFEX'
        total += self._pending_position_adjustments.get(norm_code, 0.0)
        return total

    def _resolve_inst_id(self, inst: str) -> InstrumentId:
        if "." in inst:
            return InstrumentId.from_str(inst)
        return InstrumentId(Symbol(inst), self._venue)

    def _is_option_expired(self, code: str, ts_ns: int) -> bool:
        """
        判断期权是否已到期。

        依赖 engine 传入的 _option_expiry_dates（add_option_contract 注册时填充）。
        非期权合约或未注册的合约返回 False。
        """
        if not self._option_expiry_dates or ts_ns is None:
            return False
        clean = code.split('.')[0] if '.' in code else code
        exp_day = self._option_expiry_dates.get(clean)
        if not exp_day:
            return False
        current = datetime.fromtimestamp(ts_ns / 1e9, tz=self._tz).strftime('%Y-%m-%d')
        return current > exp_day

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
