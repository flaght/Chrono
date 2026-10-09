"""EMA可恢复状态；和账户、订单关联保存在同一个CAS检查点。"""

from decimal import Decimal
import math

class AnchoredEma:
    """以已保存EMA值为递推起点，不将它当作新信号或历史成交。"""
    def __init__(self, period, count, value):
        self.period, self.count = period, count
        self.alpha = 2.0 / (period + 1.0)
        # 恢复不是一笔行情：直接保存原double，避免首笔加权再次舍入。
        self._value = float(value)

    @property
    def value(self):
        return self._value

    @property
    def initialized(self):
        return self.count >= self.period

    def update_raw(self, value):
        value = float(value)
        if not self.count:
            self._value = value
        # 与Bomber ExponentialMovingAverage.update_raw运算顺序一致。
        self._value = self.alpha * value + ((1.0 - self.alpha) * self._value)
        self.count += 1


class EmaCheckpoint:
    def __init__(self, strategy, instrument):
        self.strategy = strategy
        self.instrument = str(instrument)

    def config(self):
        c = self.strategy.config
        return {"product": c.product, "venue": c.venue, "fast": c.fast_period,
                "slow": c.slow_period, "quantity": str(c.quantity), "target_key": c.target_key}

    def snapshot_state(self):
        s = self.strategy
        return {"version": 1, "instrument": self.instrument, "config": self.config(),
                "fast": repr(s.fast.value), "slow": repr(s.slow.value),
                "bars_used": s.bars_used, "last_processed_ns": s.last_processed_ns,
                "last_main": s.last_main, "last_target": None if s.last_target is None else str(s.last_target),
                "revision": s._revision, "order_updates_received": s.order_updates_received,
                "fills_received": s.fills_received}

    def warm_closed(self, timestamp, adjusted_close):
        """补齐完整历史分钟，仅推进指标；保留最后已发布目标及其revision。"""
        s = self.strategy
        price = Decimal(str(adjusted_close))
        if timestamp <= s.last_processed_ns or not price.is_finite() or price <= 0:
            raise ValueError("补齐分钟乱序或复权价无效")
        s.fast.update_raw(float(price))
        s.slow.update_raw(float(price))
        s.bars_used += 1
        s.last_processed_ns = timestamp
        s.last_main = self.instrument.split(".")[0].lower()

    def validate_targets(self, store):
        s = self.strategy
        target = store.get(s.strategy_id)
        if target is None:
            if s._revision or s.last_target is not None:
                raise ValueError("EMA断点缺少对应已发布目标")
        elif (target.revision != s._revision
                or target.targets.get(s.config.target_key) != s.last_target):
            raise ValueError("EMA目标/版本与统一TargetStore检查点不一致")

    def restore_state(self, state):
        if state.get("version") != 1 or state.get("config") != self.config() or state.get("instrument") != self.instrument:
            raise ValueError("EMA检查点配置或真实合约不一致，须人工处理换约")
        count, revision = state["bars_used"], state["revision"]
        stamp = state["last_processed_ns"]
        fast, slow = float(state["fast"]), float(state["slow"])
        target = None if state["last_target"] is None else Decimal(state["last_target"])
        if (not isinstance(count, int) or count < 0 or not isinstance(revision, int) or revision < 0
                or not isinstance(stamp, int) or (count and stamp < 0)
                or not all(math.isfinite(v) for v in (fast, slow))
                or target not in (None, self.strategy.config.quantity, -self.strategy.config.quantity)
                or (target is not None and (count < self.strategy.config.slow_period or revision < 1))):
            raise ValueError("EMA检查点计数、数值或目标无效")
        for field in ("order_updates_received", "fills_received"):
            if not isinstance(state[field], int) or state[field] < 0:
                raise ValueError("EMA回调计数无效")
        s = self.strategy
        s.fast = AnchoredEma(s.config.fast_period, count, fast)
        s.slow = AnchoredEma(s.config.slow_period, count, slow)
        s.bars_used, s.last_processed_ns, s.last_main = count, stamp, state["last_main"]
        s.last_target, s._revision = target, revision
        s.order_updates_received, s.fills_received = state["order_updates_received"], state["fills_received"]
