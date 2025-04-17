import numpy as np
import pandas as pd

def create_data(date_index, codes, name):
    data = np.random.rand(len(date_index), len(codes))
    data = pd.DataFrame(index=date_index, columns=codes, data=data)
    data = data.stack()
    data.name = name
    return data

def create_factors(start_date, end_date, m=30, n=10, res_name=None):
    date_index = pd.date_range(start=start_date, end=end_date)
    date_index.name = 'trade_date'
    codes = ["code_" + str(i) for i in range(0, n)]

    factors_res = [
        create_data(date_index=date_index,
                    codes=codes,
                    name="data{}".format(str(i))) for i in range(0, m)
    ]
    if isinstance(res_name, str):
        factors_res.append(
            create_data(date_index=date_index, codes=codes, name=res_name))
    factors_data = pd.concat(factors_res, axis=1)

    factors_data = factors_data.reset_index().rename(
        columns={'level_1': 'code'})
    return factors_data