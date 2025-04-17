from ultron.factor.fitness.metrics import Metrics
from ultron.factor.fitness.state import State


def long_metrics(dummy_fst, yields_data, factor_data, name):
    results = Metrics.general(factors=factor_data,
                              returns=yields_data,
                              dummy=dummy_fst,
                              hold=1,
                              is_series=False)
    state = State.general(factors=factor_data, dummy=dummy_fst)
    
    st_dict = {}
    st_dict['name'] = name

    st = results.long_evaluate._asdict()
    st_dict.update(st)
    st_dict['bias'] = results.bias
    st_dict.update(state._asdict())

    return st_dict