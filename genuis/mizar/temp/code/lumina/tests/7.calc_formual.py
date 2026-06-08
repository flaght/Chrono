# -*- encoding:utf-8 -*-
import os, sys, pdb, re
import pandas as pd
import numpy as np
from collections import defaultdict
from create_data import load_random_data


sys.path.insert(0, os.path.abspath('../'))

from lumina.formual.base import FormualBase
#import lumina.impulse as impulse

formual_base = FormualBase(task_id = '20241113001', n_job = 1)


'''
s1 = formual_base.dependencies

result = defaultdict(list)
#pattern = re.compile(r'([a-zA-Z]+\d+)_([\d]+)_([\d]+)_([\d]+)')
#pattern = re.compile(r'([a-zA-Z]+\d+)_(\d+)_(\d+)_(\d+)(?:_(\d+))?')
pattern = re.compile(r'([a-zA-Z]+\d+)_(\d+)_(\d+)(?:_(\d+))?(?:_(\d+))?')
for item in s1:
    # 使用正则表达式提取信息
    match = pattern.match(item)
    if match:
        prefix = match.group(1)
        numbers = tuple(int(num) for num in match.groups()[1:] if num is not None)
        result[prefix].append(numbers)
    else:
        print('No match:', item)

pdb.set_trace()
# 将结果转换为所需的格式

def create_data():
    columns = ['close','low','high','open','volume','value','openint','chg', 'price']
    data = load_random_data(ticker_dim=4, factors_dim=len(columns) - 1, res_name=None)
    data = data.set_index(['trade_time', 'code'])
    data.columns = columns
    return data.unstack()

# 将结果转换为所需的格式
final_result = [{key: value} for key, value in result.items()]
print(final_result)
data = create_data()
res = []
for f in final_result:
    print(f)
    if isinstance(f, dict):
        name = list(f.keys())[0]
        name = "Impulse{0}".format(name.capitalize())
        cls = getattr(impulse, name)
        params = cls.serializ(list(f.values())[0])
        obj = cls(keys=params)
    else:
        cls = getattr(impulse, f)
        obj = cls()
    res += obj.calc_impulse(data).values()
data = pd.concat(res,axis=1)
pdb.set_trace()
data = data.sort_index()
data = data.reset_index()
formual_base.batch(data)
'''