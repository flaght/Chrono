from datetime import datetime
from decimal import Decimal
from math import isfinite
from zoneinfo import ZoneInfo
from bomber.framework.trader.template import StrategyTemplate
if __package__ in (None, ""):
    from delta_signal import select
else:
    from .delta_signal import select


class DeltaSelectionRecorder(StrategyTemplate):
    """选约记录基类，交易策略复用其分钟同步和审计能力。"""
    def __init__(self,strategy_id,config,options,futures,callback=None):
        super().__init__(strategy_id)
        self.config=config
        self.options=options
        self.futures=futures
        self.callback=callback
        self.latest={}
        self.pending_ns=None
        self.last_ns=-1
        self.slots=set()
        self.records=[]
        self.selections=[]
        self.audits=[]

    def on_custom_bar(self,data_key,event):
        """缓存同一分钟快照，进入下一分钟后再评价完整帧。"""
        stamp=event.ts_event
        if stamp<self.last_ns: raise ValueError("行情时间倒退")
        if stamp % 60_000_000_000:
            raise ValueError("分钟快照时间必须对齐整分钟")
        if self.pending_ns is not None and stamp>self.pending_ns:
            self._evaluate(self.pending_ns,stamp)
        if data_key in self.latest and self.latest[data_key]["event_ns"]==stamp:
            raise ValueError(f"同一分钟重复行情: {data_key}")
        self.last_ns=stamp
        self.pending_ns=stamp
        self.latest[data_key]={**event.factors,"event_ns":stamp}

    def _evaluate(self,stamp,observed_ns):
        """只读取信号分钟的报价，保存选约及候选拒绝原因。"""
        now=datetime.fromtimestamp(stamp/1e9,ZoneInfo("Asia/Shanghai"))
        key=(now.date(),now.strftime("%H:%M"))
        if key[1] not in self.config.select_slots or key in self.slots: return
        self.slots.add(key)
        if observed_ns-stamp>self.config.max_quote_age_seconds*1e9:
            self.records.append(dict(signal_ns=stamp,date=str(key[0]),slot=key[1],status="LATE_FRAME",count=0))
            return
        def current_price(symbol):
            row=self.latest.get(symbol)
            return row["close"] if row and row["event_ns"]==stamp else None
        index=current_price(self.config.index_code)
        forwards={month:current_price(symbol) for month,symbol in self.futures.items()}
        # 一致时点：只取本帧收盘价，不用旧帧价格或下一帧价格补缺。
        quotes={symbol:row for symbol,row in self.latest.items() if symbol in self.options and row["event_ns"]==stamp}
        chosen,audit,month=select(self.options,quotes,index,forwards,now,self.config)
        status="SELECTED" if len(chosen)==len(self.config.sides) else "PARTIAL" if chosen else "NO_SELECTION"
        record=dict(signal_ns=stamp,observed_ns=observed_ns,date=str(key[0]),slot=key[1],month=month,status=status,count=len(chosen))
        self.records.append(record)
        self.selections.extend({**record,**row} for row in chosen)
        self.audits.extend({**record,**row} for row in audit)
        if self.callback is not None: self.callback(record,chosen)
        self._after_selection(record, chosen, observed_ns)

    def _after_selection(self, record, chosen, request_ns):
        """记录基类不执行交易；由下层交易策略处理完整双腿结果。"""

    def on_stop(self):
        # 选约报告可在结束时处理最后一帧，因为不产生任何交易委托。
        if self.pending_ns is not None: self._evaluate(self.pending_ns,self.pending_ns)


class SnapshotParser:
    """期权先发严格原生 Bar 推进撮合，再发同分钟 CustomBar 供选约。"""
    def __init__(self, instrument_id, *, execution=False):
        self.instrument_id = instrument_id
        self.execution_parser = None
        if execution:
            from bomber.framework.market.replay.parsers.bar.fixed import FixedInstrumentBarParser
            self.execution_parser = FixedInstrumentBarParser(instrument_id,
                timestamp="datetime", available_nanoseconds="available_ns")

    def reset(self):
        if self.execution_parser is not None:
            self.execution_parser.reset()

    def parse(self, row, context):
        from math import isfinite
        from bomber.framework.market.basic.base import DataType, make_bar, make_custom_bar
        from bomber.framework.market.replay.parsers.base import ParsedEvent

        ns = int(row["datetime"].value)
        available_ns = int(row.get("available_ns", ns))
        if available_ns < ns:
            raise context.error("快照可用时间早于分钟完成时间")
        fields = ("close", "open_interest", "volume", "bar_ns", "input_quality", "zero_volume_bar")
        factors = {name: row[name] for name in fields if name in row}
        if self.execution_parser is not None:
            for parsed in self.execution_parser.parse(row, context):
                # 原生行情只送撮合一次；CustomBar 仅携带选约因子。
                yield parsed
                yield ParsedEvent(DataType.CUSTOM_BAR, self.instrument_id,
                    make_custom_bar(parsed.payload, factors), "1-MINUTE")
            return
        # 原生 Bar 仅承载事件；原始 close 保留在 factors 中，不把替代值送给选约。
        carrier = float(row["close"])
        if not isfinite(carrier) or carrier <= 0:
            carrier = 1.0
        bar = make_bar(self.instrument_id, carrier, carrier, carrier, carrier, 0, ns, available_ns,
                       meta=context.require_meta(self.instrument_id), bar_type="1-MINUTE")
        yield ParsedEvent(DataType.CUSTOM_BAR, self.instrument_id, make_custom_bar(bar, factors), "1-MINUTE")


class OptionDeltaStrategy(DeltaSelectionRecorder):
    """买入完整 Call/Put 组合，先平旧组合再开新组合，期末真实清仓。"""
    def __init__(self, strategy_id, config, options, futures, *, instruments, final_day):
        if set(config.sides) != {"C", "P"}:
            raise ValueError("双腿交易必须同时启用 C 和 P")
        super().__init__(strategy_id, config, options, futures)
        self.instruments = dict(instruments)
        self.final_day = final_day
        self.goals = {key: Decimal(0) for key in instruments}
        self.pending_targets = None
        self.pending_reason = None
        self.signals = []
        self.submissions = 0
        self.flatten_requested = False
        self.waiting_for = None

    def _evaluate(self, stamp, observed_ns):
        now = datetime.fromtimestamp(stamp / 1e9, ZoneInfo("Asia/Shanghai"))
        # 隔夜或长缺口不能把最后一帧信号变成下一场次的订单。
        if observed_ns - stamp > self.config.max_quote_age_seconds * 1_000_000_000:
            super()._evaluate(stamp, observed_ns)
            return
        request_time = datetime.fromtimestamp(observed_ns / 1e9, ZoneInfo("Asia/Shanghai"))
        final_flat = request_time.date() == self.final_day and request_time.strftime("%H:%M") >= self.config.flatten_slot
        exposed = [key for key, item in self.instruments.items()
                   if self.account_position(str(item)) or self.goals[key] or self.working_quantity(str(item))
                   or (self.pending_targets is not None and self.pending_targets[key])]
        for key in exposed:
            if now.date() > self.options[key].expiry:
                raise RuntimeError(f"{key}已过最后交易日，不能继续保留真实仓位或目标")
        expiry_flat = any((self.options[key].expiry - now.date()).days <= self.config.close_remaining_days
                          for key in exposed)
        if final_flat or expiry_flat:
            self.flatten_requested |= final_flat
            self.pending_targets = {key: Decimal(0) for key in self.instruments}
            self.pending_reason = "FINAL_FLAT" if final_flat else "PRE_EXPIRY_FLAT"
        if not final_flat:
            super()._evaluate(stamp, observed_ns)
        else:
            slot = (now.date(), now.strftime("%H:%M"))
            if slot[1] in self.config.select_slots and slot not in self.slots:
                self.slots.add(slot)
                self.records.append(dict(signal_ns=stamp, date=str(slot[0]), slot=slot[1],
                    status="FINAL_EXIT_ONLY", count=0))
        self._drive_targets(stamp, observed_ns)

    def _after_selection(self, record, chosen, request_ns):
        if record["status"] != "SELECTED" or {row["kind"] for row in chosen} != {"C", "P"}:
            self.signals.append({"signal_ns": record["signal_ns"], "request_ns": request_ns,
                "action": "NO_PAIR_KEEP_POSITION", "selection_status": record["status"]})
            return
        desired = {key: Decimal(0) for key in self.instruments}
        for row in chosen:
            if row["symbol"] not in desired:
                raise RuntimeError(f"选中合约未注册交易: {row['symbol']}")
            desired[row["symbol"]] = Decimal(self.config.quantity)
        # 替换整个组合目标；每日相同合约不会累加手数。
        self.pending_targets = desired
        self.pending_reason = "BUY_PAIR"
        self.waiting_for = None

    def _drive_targets(self, signal_ns, request_ns):
        if self.pending_targets is None:
            return
        actual = {key: self.account_position(str(item)) for key, item in self.instruments.items()}
        if any(self.working_quantity(str(item)) for item in self.instruments.values()):
            return  # 已提交订单尚在途，等待成交后再推进换仓。
        desired = self.pending_targets
        if any(position and not desired[key] for key, position in actual.items()):
            stage = {key: Decimal(0) for key in self.instruments}
            reason = "CLOSE_OLD_PAIR" if any(desired.values()) else self.pending_reason
        else:
            stage, reason = desired, self.pending_reason
        affected = [key for key in stage if actual[key] != stage[key] or self.goals[key] != stage[key]]
        if not affected:
            if stage == desired:
                self.pending_targets = None
            return
        # 下单只依赖本次变化的腿；未持有的历史合约不要求报价。
        missing = []
        for key in affected:
            row = self.latest.get(key)
            if (row is None or row["event_ns"] > signal_ns
                    or not 0 <= request_ns - row["event_ns"] <= self.config.max_market_age_seconds * 1_000_000_000
                    or not isfinite(row["close"]) or row["close"] <= 0):
                missing.append(key)
        if missing:
            waiting = (reason, tuple(sorted(missing)))
            if waiting != self.waiting_for:
                self.signals.append({"signal_ns": signal_ns, "request_ns": request_ns,
                    "action": "WAIT_FRESH_PRICES", "reason": reason, "missing": missing})
                self.waiting_for = waiting
            return
        targets = {str(self.instruments[key]): value for key, value in stage.items()}
        self.set_targets(targets, request_ns, metadata={"signal_ts": signal_ns, "action": reason})
        self.goals = dict(stage)
        self.submissions += 1
        self.waiting_for = None
        self.signals.append({"signal_ns": signal_ns, "request_ns": request_ns, "action": reason,
            "targets": {key: str(value) for key, value in targets.items() if value}})
        # 等实际成交后再清除待执行计划，不能把提交目标当作成交完成。

    def on_stop(self):
        """结束时不补发订单，最后一帧之后没有行情可用于撮合。"""
        if self.pending_ns is not None:
            now = datetime.fromtimestamp(self.pending_ns / 1e9, ZoneInfo("Asia/Shanghai"))
            slot = (now.date(), now.strftime("%H:%M"))
            if slot[1] in self.config.select_slots and slot not in self.slots:
                self.slots.add(slot)
                self.records.append(dict(signal_ns=self.pending_ns, date=str(slot[0]), slot=slot[1],
                    status="NO_FOLLOWING_FRAME", count=0))

    def on_order_update(self, event):
        status = getattr(event.status, "value", str(event.status))
        if status in ("REJECTED", "CANCELED", "EXPIRED"):
            raise RuntimeError(f"订单{status}: {event.identity.instrument_id} {event.reason}")
