import itertools, pdb, re, os
import pandas as pd
from ultron.sentry.api import *
from collections import defaultdict
from .process import split_k, create_parellel, create_factors
from .impulse import Impulse


class FormualBase(object):

    def __init__(self, task_id, n_job=1, impulse=None, **kwargs):
        __str__ = 'factor_formual_{}'.format(task_id)
        self.category = 'ff' if 'category' not in kwargs else kwargs['category']
        self.name = '公式因子' if 'name' not in kwargs else kwargs['name']
        self.task_id = task_id
        self.n_job = n_job
        self.init_formual(task_id, **kwargs)
        self.impulse = Impulse(
            self.dependencies) if impulse is None else impulse

    def init_formual(self, task_id, **kwargs):
        if 'formual' in kwargs:
            formual_data = pd.DataFrame(kwargs['formual'])
        else:
            filename = "{0}.feather".format(
                task_id
            ) if 'LUMINA_DATA_PATH' not in os.environ else os.path.join(
                os.environ['LUMINA_DATA_PATH'], "{0}.feather".format(task_id))
            formual_data = pd.read_feather(filename)
        dependencies = [
            eval(v['formual'])._dependency
            for v in formual_data.to_dict(orient='records')
        ]
        dependencies = list(set(itertools.chain.from_iterable(dependencies)))

        max_window = max([
            eval(f['formual'])._window
            for f in formual_data.to_dict(orient='records')
        ])
        self.formual_data = formual_data
        self._max_window = max_window
        self._dependencies = dependencies

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def max_window(self):
        return self._max_window

    def batch(self, data=None, method='impulse'):
        if method == 'impulse':
            return self.impulse.batch(data)
        else:
            data = data.sort_values(
                by=['trade_time', 'code']).set_index('trade_time')
            target_columns = self.formual_data.to_dict(orient='records')
            process_list = split_k(self.n_job, target_columns)
            res = create_parellel(process_list=process_list,
                                  callback=create_factors,
                                  basic_data=data)
            res = list(itertools.chain.from_iterable(res))
            factors_data = pd.concat(res, axis=1)
            return factors_data.reset_index()
