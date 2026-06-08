### 二次封装，主要是通过设置method 和 window， 同时考虑自定义实现
import inspect
from functools import wraps
#import ultron.factor.empyrical as base_empyrical
import empyrical as base_empyrical
from ultron.utilities.logger import kd_logger


class Empyrical(object):

    def __init__(self):
        self._registry = {}
        self._register_default_methods()

    def _register_default_methods(self):
        default_methods = [
            # 无需 window 的函数
            ('sharpe_ratio', base_empyrical.sharpe_ratio),
            ('calmar_ratio', base_empyrical.calmar_ratio),
            ('annual_return', base_empyrical.annual_return),
            ('cum_returns_final', base_empyrical.cum_returns_final),
            ('max_drawdown', base_empyrical.max_drawdown),

            # 需要 window 的函数
            ('roll_sharpe_ratio', base_empyrical.roll_sharpe_ratio),
            ('roll_max_drawdown', base_empyrical.roll_max_drawdown),
            ('roll_sortino_ratio', base_empyrical.roll_sortino_ratio)
        ]
        for name, func in default_methods:
            self.register_method(name, func)

    def register_method(self, name: str, func: callable):
        self._registry[name] = func
        kd_logger.debug(f"方法 '{name}' 已成功注册。")

    def list_methods(self):
        for name in sorted(self._registry.keys()):
            kd_logger.info(f"- {name}")

    def calculate(self, returns_series, method, window=None, period='daily'):
        if method not in self._registry:
            raise ValueError(
                f"方法 '{method}' 未注册。可用方法: {list(self._registry.keys())}")

        func = self._registry[method]
        kwargs = {}
        func_params = inspect.signature(func).parameters
        period_map = {
            'daily': base_empyrical.DAILY,
            'weekly': base_empyrical.WEEKLY,
            'monthly': base_empyrical.MONTHLY
        }
        if 'period' in func_params:
            kwargs['period'] = period_map.get(period.lower(), base_empyrical.DAILY)

        if 'window' in func_params:
            if window is None:
                # 如果函数需要 window 但未提供，则引发错误
                raise ValueError(f"方法 '{method}' 需要一个 'window' 参数，但未提供。")
            kwargs['window'] = window

        return func(returns_series, **kwargs)


empyrical = Empyrical()