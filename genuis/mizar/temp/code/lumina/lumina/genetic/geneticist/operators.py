from ultron.factor.genetic.geneticist.operators import Operators as BaseOperators
from ultron.factor.genetic.geneticist.operators import FunctionType, Function
import pdb

class Operators(object):

    def __init__(self):
        self.base_operators = BaseOperators()

    @property
    def cross_section_mutation(self):
        return self.base_operators._cross_section_mutation_list

    @property
    def cross_section_crossover(self):
        return self.base_operators._cross_section_crossover_list

    @property
    def time_series_mutation(self):
        return self.base_operators._time_series_mutation_list

    @property
    def time_series_crossover(self):
        return self.base_operators._time_series_crossover_list
    

    def create_operators(self, operators_sets, period_params):
        ## 截面变异算子
        cs_mutation_function = [
            Function(f, 1, FunctionType.cross_section)
            for f in self.cross_section_mutation if f.__name__ in operators_sets
        ]
        ## 截面交叉算子
        cross_crossover_function = [
            Function(f, 2, FunctionType.cross_section)
            for f in self.cross_section_crossover if f.__name__ in operators_sets
        ]
        ## 时序变异算子
        time_mutation_function = [
            Function(f, 1, FunctionType.time_series, period)
            for f in self.time_series_mutation for period in period_params if f.__name__ in operators_sets
        ]

        ## 时序交叉算子
        time_crossove_function = [
            Function(f, 1, FunctionType.time_series, period)
            for f in self.time_series_crossover for period in period_params if f.__name__ in operators_sets
        ]
        return cs_mutation_function + cross_crossover_function + time_mutation_function + time_crossove_function 
