from collections import namedtuple


class StrategyTuple(
        namedtuple('StrategyTuple',
                   ('name', 'formual', 'signal_method', 'signal_params',
                    'strategy_method', 'strategy_params', 'fitness'))):
    __slots__ = ()


class EmpyricalTuple(
        namedtuple('EmpyricalTuple',
                   ('name', 'annual_return', 'annual_volatility', 'calmar',
                    'sharpe', 'max_drawdown', 'sortino', 'returns_series'))):
    __slots__ = ()

    def dumps(self):
        return {
            'name': self.name,
            'annual_return': self.annual_return,
            'annual_volatility': self.annual_volatility,
            'calmar': self.calmar,
            'sharpe': self.sharpe,
            'max_drawdown': self.max_drawdown,
            'sortino': self.sortino
        }


class EmpyoicalTuple(
        namedtuple('FusionTuple',
                   ('name', 'win_rate', 'profit_rate', 'profit_std'))):
    __slots__ = ()

    def dumps(self):
        return self._asdict()


class AssignmentTuple(
        namedtuple('AssignmentTuple', ('name', 'params', 'cluster'))):
    __slots__ = ()


class KMeansResultTuple(
        namedtuple('KMeansResultTuple',
                   ('name', 'params', 'cluster', 'mapping', 'empyrical'))):
    __slots__ = ()
