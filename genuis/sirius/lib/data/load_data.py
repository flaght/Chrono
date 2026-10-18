import pdb, os
import pandas as pd
from jdw import DBAPI
from alphacopilot.api.data import RetrievalAPI, ddb_tools, DDBAPI

kd_engine = DBAPI.FetchEngine.create_engine('kd')


def fetch_algin_factors(begin_date, end_date, codes=None, columns=None):
    name = 'fut_algin_factors'
    names = DBAPI.CustomizeFactory(kd_engine).name(name=name)
    clause_list = [
        names.trade_date >= begin_date, names.trade_date <= end_date
    ]
    if isinstance(codes, list):
        clause_list.append(names.code.in_(codes))
    algin_factors_data = DBAPI.CustomizeFactory(kd_engine).custom(
        name=name, clause_list=clause_list, columns=columns)
    return algin_factors_data


def fetch_tick_data(base_path, begin_date, end_date, codes):
    algin_factors = fetch_algin_factors(
        begin_date=begin_date,
        end_date=end_date,
        codes=codes,
        columns=['trade_date', 'code', 'symbol'])
    algin_factors = algin_factors.drop_duplicates(
        subset=['trade_date', 'code'], keep='last')

    file_res = []
    for row in algin_factors.itertuples():
        filename1 = os.path.join(base_path, 'data','main_tick', row.code, "{0}.feather".format(
            row.trade_date.strftime('%Y%m%d')))
        tick_data = pd.read_feather(filename1)
        print(filename1)
        file_res.append(tick_data)
    return pd.concat(file_res,axis=0)
