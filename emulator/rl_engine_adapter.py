# rl_engine_adapter.py
import types, pdb
from cn_futures import CNFutures
from rl_strategy_adapter import RLStrategyAdapter
from object import BarData, Interval # 需要导入以创建 dummy bar

class EngineAdapter:
    """
    适配器，在不修改 CNFutures 源码的情况下，
    使其能够支持RL所需的单步（step-by-step）执行模式。
    """
    def __init__(self, code: str, uri: str, rl_strategy: RLStrategyAdapter):
        self.code = code
        self.uri = uri
        self.strategy = rl_strategy
        self.engine: CNFutures = None
        self._execution_generator = None
        
        # 保存原始方法的字典，用于恢复
        self._original_methods = {}

    def reset(self, trade_date: str):
        """重置引擎以开始新的一天，并返回第一个 bar 数据和结束标志。"""
        self.strategy.reset_for_new_episode()
        
        self.engine = CNFutures(
            code=self.code, 
            uri=self.uri,
            strategies_pool={'rl_agent': self.strategy}
        )
        
        self._execution_generator = self._run_day_generator(trade_date)
        
        # 预跑引擎，直到第一个 on_bar 完成并 yield
        return self._advance_to_next_bar()

    def step(self):
        """驱动引擎执行到下一个 on_bar 完成并暂停。"""
        return self._advance_to_next_bar()

    def _advance_to_next_bar(self):
        """从生成器中获取下一个bar，恢复引擎的执行。"""
        try:
            bar = next(self._execution_generator)
            return bar, False # (bar_data, is_day_end)
        except StopIteration:
            return None, True

    def _patch_methods(self):
        """动态替换(Monkey Patch)引擎和策略的方法以植入yield。"""
        engine = self.engine
        
        # --- 保存原始方法 ---
        self._original_methods = {
            'strategy_on_bar': self.strategy.on_bar,
            'engine_create_bar': engine.create_bar,
            'engine_on_tick': engine.on_tick
        }

        # --- 定义新的、可yield的方法 ---
        def yielding_strategy_on_bar(strategy_self, bar):
            self._original_methods['strategy_on_bar'](bar) # 调用原始的结算、更新等
            yield bar # 暂停并返回 bar

        # 重新实现 create_bar 和 on_tick，使其能够传递返回值
        def yielding_engine_create_bar(engine_self, market_tick):
            tick_minute = int(market_tick.create_time.minute)
            if tick_minute != engine_self.bar_mintue:
                if engine_self.market_bar:
                    engine_self.market_bar.create_time = engine_self.market_bar.create_time.floor('min')
                    for _, strategy in engine_self.strategies_pool.items():
                        return strategy.on_bar(engine_self.market_bar)
                
                engine_self.market_bar = BarData(
                    code=market_tick.code, symbol=market_tick.symbol, exchange=market_tick.exchange,
                    create_time=market_tick.create_time, interval=Interval.MINUTE, volume=market_tick.volume,
                    turnover=market_tick.turnover, open_interest=market_tick.open_interest,
                    open_price=market_tick.last_price, high_price=market_tick.last_price,
                    low_price=market_tick.last_price, close_price=market_tick.last_price
                )
                engine_self.bar_mintue = tick_minute
            else:
                engine_self.market_bar.high_price = max(engine_self.market_bar.high_price, market_tick.last_price)
                engine_self.market_bar.low_price = min(engine_self.market_bar.low_price, market_tick.last_price)
                engine_self.market_bar.close_price = market_tick.last_price
                engine_self.market_bar.volume = market_tick.volume
                engine_self.market_bar.open_interest = market_tick.open_interest
                engine_self.market_bar.turnover = market_tick.turnover
                engine_self.market_bar.create_time = market_tick.create_time
        
        def yielding_engine_on_tick(engine_self, trade_date, market_tick):
            engine_self._recovery_to_working()
            engine_self._cross_limit_order(market_tick)
            for name, strategy in engine_self.strategies_pool.items():
                strategy.on_tick(market_tick)
            engine_self.market_tick = market_tick
            return engine_self.create_bar(market_tick=market_tick)

        # --- 应用补丁 ---
        self.strategy.on_bar = types.MethodType(yielding_strategy_on_bar, self.strategy)
        engine.create_bar = types.MethodType(yielding_engine_create_bar, engine)
        engine.on_tick = types.MethodType(yielding_engine_on_tick, engine)

    def _unpatch_methods(self):
        """恢复所有被替换的原始方法。"""
        if not self._original_methods: return
        self.strategy.on_bar = self._original_methods['strategy_on_bar']
        self.engine.create_bar = self._original_methods['engine_create_bar']
        self.engine.on_tick = self._original_methods['engine_on_tick']
        self._original_methods.clear()

    def _run_day_generator(self, trade_date: str):
        """生成器函数，封装 CNFutures 的日内循环。"""
        self._patch_methods()
        engine = self.engine
        
        # --- 模拟 cn_futures.start() 的核心循环 ---
        try:
            market_tick_list = engine.load_tick(trade_date=trade_date)
            market_daily = engine.load_daily(trade_date=trade_date)
            
            for market_tick in market_tick_list:
                if market_tick.volume == 0 or market_tick.last_price >= market_tick.limit_up or market_tick.last_price <= market_tick.limit_down:
                    continue

                result_generator = engine.on_tick(trade_date=trade_date, market_tick=market_tick)
                
                if isinstance(result_generator, types.GeneratorType):
                    yield from result_generator
            
            # --- 日终处理 ---
            recovery_oids = list(engine.recovery_limit_order.keys())
            for oid in recovery_oids:
                if oid in engine.recovery_limit_order:
                    order = engine.recovery_limit_order.pop(oid)
                    order.status = engine.OrderStatus.REJECTED
                    engine.strategies_pool[order.strategy_id].on_order(order)
            
            for _, strategy in engine.strategies_pool.items():
                strategy.after_market_close(trade_date, market_daily)

        finally:
            # 确保无论发生什么，原始方法都会被恢复
            self._unpatch_methods()