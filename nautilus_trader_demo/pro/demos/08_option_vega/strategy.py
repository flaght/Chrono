"""认购期权卖方与同月期货对冲：管理真实持仓、在途订单和清仓。"""
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from bomber.framework.trader.template import StrategyTemplate
from .vega_signal import remaining_trading_days, tau, allocate_short, cached_greeks, implied_greeks


class OptionVegaStrategy(StrategyTemplate):
    """纯数值函数负责选约配仓，策略负责行情时序和实际敞口执行。"""
    def __init__(self, strategy_id, config, contracts, futures, instruments,
                 index_id, trading_days, final_day=None, options_by_day=None):
        super().__init__(strategy_id)
        self.config = config
        self.contracts = dict(contracts)
        self.futures = dict(futures)
        self.instruments = dict(instruments)
        self.index_id = index_id
        self.trading_days = tuple(trading_days)
        self.final_day = final_day
        self.options_by_day = options_by_day or {}
        self.restored = False
        self.stopping = False
        self.order_updates_received = 0
        self.fills_received = 0
        self.latest = {}
        self.pending_ts = None
        self.resume_ns = None
        self.last_event_ns = -1
        self.option_goals = {key: Decimal(0) for key in self.contracts}
        self.future_goals = {key: Decimal(0) for key in self.futures}
        self.iv_cache = {}
        self.sent_days = set()
        self.last_actual = None
        self.submissions = 0
        self.entry_decisions = 0
        self.entry_audit = []
        self.frames = 0
        self.signals = []
        self.flatten_requested = False

    def on_bar(self, data_key, bar):
        """进入新分钟时处理前一完整分钟，确保决策不读取尚未送达的行情。"""
        timestamp = bar.ts_event
        if timestamp < self.last_event_ns:
            raise ValueError("行情时间倒退")
        if self.pending_ts is not None and timestamp > self.pending_ts:
            # 此时上一分钟全体行情已到齐；新分钟行情尚未加入策略缓存。
            if timestamp - self.pending_ts <= self.config.max_market_age_seconds * 1_000_000_000:
                self._frame(self.pending_ts, timestamp)
            else:
                self.resume_ns = timestamp
                self.signals.append({"signal_ns": self.pending_ts, "request_ns": timestamp,
                                     "action": "GAP_FRAME_SKIPPED"})
        self.last_event_ns = timestamp
        self.pending_ts = timestamp
        if data_key in self.latest and self.latest[data_key][0] == timestamp:
            raise ValueError(f"同一分钟重复行情: {data_key}")
        self.latest[data_key] = (timestamp, float(bar.close.as_decimal()))

    def on_market_frame(self, frame):
        """所有输入已封闭；来源和时间适配由市场层完成，不读接收时钟估值。"""
        timestamp = frame.valuation_ns
        if timestamp <= self.last_event_ns:
            raise ValueError("行情帧时间倒退或重复")
        if self.last_event_ns >= 0 and timestamp - self.last_event_ns > self.config.max_market_age_seconds * 1_000_000_000:
            self.resume_ns = timestamp
        if self.restored:
            # 恢复已成交事实，不继续旧会话尚未完成的开仓目标。
            self.option_goals = {key: self.position(str(self.instruments[key])) for key in self.contracts}
            self.future_goals = {key: self.position(str(self.instruments[key])) for key in self.futures}
            self.resume_ns = timestamp
            self.restored = False
        self.last_event_ns = timestamp
        for key, observation in frame.observations.items():
            self.latest[key] = (timestamp, float(observation.price))
        self._frame(timestamp, frame.request_ns)

    def required_market_keys(self):
        required = {key for key, instrument in self.instruments.items()
            if self.position(str(instrument)) or self.strategy_working_quantity(str(instrument))
            or self.option_goals.get(key) or self.future_goals.get(key)}
        held = required.intersection(self.contracts)
        required.add(self.index_id)
        required.update(self.contracts[key].future for key in held)
        return required

    def on_fill(self, event):
        self.fills_received += 1

    def _remaining_days(self, day, last_day, threshold=None):
        """按已知交易日检查开仓或退出阈值，不把未来日期缺失视为临近到期。"""
        return remaining_trading_days(self.trading_days, day, last_day,
            self.config.min_remaining_days if threshold is None else threshold)

    def _price(self, symbol, signal_ns):
        """查询新鲜真实价格，缺失时附带时间、仓位和在途数量报错。"""
        value = self.latest.get(symbol)
        if value is None or not 0 <= signal_ns - value[0] <= self.config.max_market_age_seconds * 1_000_000_000:
            signal_time = datetime.fromtimestamp(signal_ns / 1e9, ZoneInfo("Asia/Shanghai"))
            last_time = None if value is None else datetime.fromtimestamp(value[0] / 1e9, ZoneInfo("Asia/Shanghai"))
            age = None if value is None else (signal_ns - value[0]) / 1e9
            instrument = self.instruments.get(symbol)
            position = None if instrument is None else self.position(str(instrument))
            working = None if instrument is None else self.strategy_working_quantity(str(instrument))
            goal = self.option_goals.get(symbol, self.future_goals.get(symbol))
            raise RuntimeError(
                f"持仓计算缺少新鲜行情: {symbol}; 决策时间={signal_time.isoformat()} "
                f"最新报价时间={None if last_time is None else last_time.isoformat()} "
                f"报价年龄秒={age} 限制秒={self.config.max_market_age_seconds} "
                f"实际仓位={position} 目标仓位={goal} 在途数量={working}"
            )
        return value[1]

    def _greeks(self, symbol, signal_ns, now):
        """以同月期货和期权价格反解风险指标，必要时使用有限期波动率缓存。"""
        info = self.contracts[symbol]
        forward = self._price(info.future, signal_ns)
        premium = self._price(symbol, signal_ns)
        years = tau(now, info.last_day)
        greeks = None
        if premium - max(forward - info.strike, 0) >= self.config.min_time_value:
            greeks = implied_greeks(forward, info.strike, years, self.config.rate, premium)
        if greeks is not None:
            self.iv_cache[symbol] = (signal_ns, greeks.iv)
        else:
            cached = self.iv_cache.get(symbol)
            if cached and signal_ns - cached[0] <= self.config.max_iv_age_seconds * 1_000_000_000:
                greeks = cached_greeks(forward, info.strike, years, self.config.rate, cached[1])
        return greeks

    def _frame(self, signal_ns, request_ns):
        """按实际仓位检查退出、每日卖方选约、期货对冲及末日清仓。"""
        self.frames += 1
        now = datetime.fromtimestamp(signal_ns / 1e9, ZoneInfo("Asia/Shanghai"))
        day, slot = now.date(), now.strftime("%H:%M")
        actual = {key: self.position(str(instrument)) for key, instrument in self.instruments.items()}
        working = {key: self.strategy_working_quantity(str(instrument)) for key, instrument in self.instruments.items()}
        current = tuple((key, actual[key]) for key in sorted(self.contracts))

        if self.resume_ns is not None:
            # 隔夜/午休恢复时，各腿首Bar可相差一分钟；等待更新，不用上一场次报价。
            required = {key for key in self.instruments if actual[key] or working[key]
                        or self.option_goals.get(key) or self.future_goals.get(key)}
            held_options = required.intersection(self.contracts)
            if held_options:
                required.add(self.index_id)
                required.update(self.contracts[key].future for key in held_options)
            missing = [key for key in sorted(required)
                       if key not in self.latest or not self.resume_ns <= self.latest[key][0] <= signal_ns
                       or signal_ns - self.latest[key][0] > self.config.max_market_age_seconds * 1_000_000_000]
            if missing:
                if signal_ns - self.resume_ns <= self.config.max_market_age_seconds * 1_000_000_000:
                    self.signals.append({"signal_ns": signal_ns, "request_ns": request_ns,
                                         "action": "RESUME_WAIT", "missing": missing})
                    return
                self._price(missing[0], signal_ns)  # 超时后输出含时间/仓位的诊断。
                raise RuntimeError(f"场次恢复后行情仍未更新: {missing}")
            self.resume_ns = None

        if day == self.final_day and slot >= self.config.flatten_slot:
            self.flatten_requested = True
            if not any(working.values()):
                targets = {str(value): Decimal(0) for value in self.instruments.values()}
                if any(actual.values()) or any(self.option_goals.values()) or any(self.future_goals.values()):
                    for key, position in actual.items():
                        if position:
                            self._price(key, signal_ns)
                    self.option_goals = {key: Decimal(0) for key in self.contracts}
                    self.future_goals = {key: Decimal(0) for key in self.futures}
                    self._submit(targets, request_ns, signal_ns, "FINAL_FLAT")
            self.last_actual = current
            return

        changed = False
        # 每分钟管理已有目标/实际持仓，禁止通过删除字典冒充平仓。
        held = [key for key in self.contracts if actual[key] or self.option_goals[key] or working[key]]
        for key in held:
            info = self.contracts[key]
            if day > info.last_day and actual[key]:
                raise RuntimeError(f"{key}已过最后交易日但仍有实际持仓")
            if not working[key]:
                close = self._price(key, signal_ns)
                index = self._price(self.index_id, signal_ns)
                should_close = (self._remaining_days(day, info.last_day, self.config.close_remaining_days + 1) <= self.config.close_remaining_days
                                or close - max(index - info.strike, 0) <= 0)
                if should_close and self.option_goals[key] != 0:
                    self.option_goals[key] = Decimal(0)
                    changed = True
            if actual[key] and self.option_goals[key] and self._greeks(key, signal_ns, now) is None:
                raise RuntimeError(f"{key}无法计算可靠Delta，停止而不撤掉保护性对冲")

        if slot == self.config.option_slot and day not in self.sent_days:
            audit = {"signal_ns": signal_ns, "request_ns": request_ns, "date": str(day),
                "slot": slot, "outcome": "WORKING_ORDERS", "candidate_count": 0, "targets": {}}
            self.entry_audit.append(audit)
            del self.entry_audit[:-32]
            if not any(working[key] for key in self.contracts):
                index = self._price(self.index_id, signal_ns)
                candidates = []
                for key in self.options_by_day.get(day, ()):
                    info = self.contracts[key]
                    if not info.list_day <= day <= info.last_day or info.strike <= index:
                        continue
                    if self._remaining_days(day, info.last_day) < self.config.min_remaining_days:
                        continue
                    # 开仓只使用期权、期货、指数同一分钟价格，不向前填充。
                    if any(self.latest.get(name, (None,))[0] != signal_ns for name in (key, info.future, self.index_id)):
                        continue
                    greeks = self._greeks(key, signal_ns, now)
                    if greeks is not None:
                        candidates.append((key, greeks, info.multiplier))
                self.sent_days.add(day)
                audit["candidate_count"] = len(candidates)
                audit["outcome"] = "NO_VALID_ENTRY"
                if candidates:
                    nearest = min(self.contracts[item[0]].last_day for item in candidates)
                    candidates = [item for item in candidates if self.contracts[item[0]].last_day == nearest]
                    goals = allocate_short(candidates, self.config.delta_targets, self.config.vega_cash_per_unit,
                                           self.config.max_option_lots, self.config.max_total_option_lots)
                    audit["outcome"] = "TARGET" if goals else "ZERO_ALLOCATION"
                    audit["targets"] = {key: str(value) for key, value in goals.items()}
                    audit["future_bindings"] = {key: self.contracts[key].future for key in goals}
                    if goals:
                        # 组合总目标替换，不将每日计算数量累加到昨日仓位。
                        self.option_goals = {key: Decimal(goals.get(key, 0)) for key in self.contracts}
                        self.entry_decisions += 1
                        changed = True
                else:
                    self.signals.append({"signal_ns": signal_ns, "request_ns": request_ns,
                                         "action": "NO_VALID_ENTRY", "date": str(day)})

        hedge_due = changed or current != self.last_actual or slot in self.config.hedge_slots
        if hedge_due and not any(working[key] for key in self.futures):
            totals = {key: 0.0 for key in self.futures}
            closing_without_iv = set()
            for key in self.contracts:
                if not actual[key]:
                    continue
                greeks = self._greeks(key, signal_ns, now)
                info = self.contracts[key]
                if greeks is None:
                    if self.option_goals[key] == 0:
                        closing_without_iv.add(info.future)
                        continue
                    raise RuntimeError(f"{key}IV无效，无法对冲实际持仓")
                totals[info.future] += float(actual[key]) * float(info.multiplier) * greeks.delta
            goals = {}
            for key, exposure in totals.items():
                if key in closing_without_iv:
                    goals[key] = self.future_goals[key]  # 先平期权，保留既有对冲直到实际平仓。
                    continue
                lots = round(-exposure / float(self.futures[key]["contMultNum"]))
                if abs(lots) > self.config.max_future_lots:
                    raise RuntimeError(f"{key}所需对冲{lots}手超过上限，请降低Vega预算")
                goals[key] = Decimal(lots)
            changed = changed or goals != self.future_goals
            self.future_goals = goals
            self.last_actual = current

        if changed:
            targets = {str(self.instruments[key]): value for key, value in
                       {**self.option_goals, **self.future_goals}.items()}
            # 显式带零目标，包括换月旧腿；组合提交不代表多腿原子成交。
            self._submit(targets, request_ns, signal_ns, "REBALANCE")

    def _submit(self, targets, request_ns, signal_ns, action):
        """提交完整目标，保留信号时间与请求时间以核对决策时序。"""
        self.set_targets(targets, request_ns, metadata={"signal_ts": signal_ns, "action": action})
        self.submissions += 1
        self.signals.append({"signal_ns": signal_ns, "request_ns": request_ns,
                             "action": action, "targets": {key: str(value) for key, value in targets.items() if value}})

    def on_order_update(self, event):
        """订单拒绝、取消或过期时停止，避免静默留下不完整对冲。"""
        self.order_updates_received += 1
        status = getattr(event.status, "value", str(event.status))
        if status == "REJECTED" or (not self.stopping and status in ("CANCELED", "EXPIRED")):
            raise RuntimeError(f"订单{status}: {event.identity.instrument_id} {event.reason}")
