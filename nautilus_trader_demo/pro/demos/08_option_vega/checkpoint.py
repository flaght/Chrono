"""Vega策略信号状态；价格、IV缓存和正在组装的行情帧不作为新鲜数据恢复。"""
from datetime import date
from decimal import Decimal


class OptionVegaCheckpoint:
    def __init__(self, strategy):
        self.strategy = strategy

    def snapshot_state(self):
        s = self.strategy
        return {"version": 1,
            "option_goals": {key: str(value) for key, value in s.option_goals.items()},
            "future_goals": {key: str(value) for key, value in s.future_goals.items()},
            "sent_days": sorted(str(day) for day in s.sent_days),
            "submissions": s.submissions, "entry_decisions": s.entry_decisions,
            "flatten_requested": s.flatten_requested}

    def restore_state(self, state):
        if state.get("version") != 1:
            raise ValueError("不支持的Vega策略检查点")
        s = self.strategy
        restored = {}
        for name, expected, cap in (("option_goals", s.contracts, s.config.max_option_lots),
                                    ("future_goals", s.futures, s.config.max_future_lots)):
            values = state[name]
            if set(values) != set(expected):
                raise ValueError("恢复时Vega合约集合不一致")
            values = {key: Decimal(value) for key, value in values.items()}
            if any(not value.is_finite() or value != value.to_integral_value() or abs(value) > cap
                   or name == "option_goals" and value > 0 for value in values.values()):
                raise ValueError("恢复目标数量无效或超过配置上限")
            restored[name] = values
        if sum(abs(v) for v in restored["option_goals"].values()) > s.config.max_total_option_lots:
            raise ValueError("恢复期权总手数超过上限")
        days = {date.fromisoformat(value) for value in state["sent_days"]}
        if not days <= set(s.trading_days):
            raise ValueError("恢复选约日期不在日历内")
        for name in ("submissions", "entry_decisions"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError("恢复计数无效")
        if type(state["flatten_requested"]) is not bool:
            raise ValueError("恢复清仓状态无效")
        s.option_goals, s.future_goals = restored["option_goals"], restored["future_goals"]
        s.sent_days = days
        s.submissions, s.entry_decisions = state["submissions"], state["entry_decisions"]
        s.flatten_requested = state["flatten_requested"]
        s.latest.clear()
        s.iv_cache.clear()
        s.pending_ts = s.resume_ns = s.last_actual = None
        s.last_event_ns = -1
        s.restored = True
