import os,pdb
import pandas as pd
from ultron.strategy.models.processing import winsorize as alk_winsorize
from ultron.strategy.models.processing import standardize as alk_standardize
from dotenv import load_dotenv
load_dotenv()
from kdutils.data import fetch_base


def normal_factors(method, horizon):
    pdb.set_trace()
    dirs = os.path.join(os.environ['BASE_PATH'], method, 'evolution', str(horizon))
    filename = os.path.join(dirs, "factors_data.feather")
    factors_data = pd.read_feather(filename)
    begin_date = factors_data['trade_date'].min().strftime('%Y-%m-%d')
    end_date = factors_data['trade_date'].max().strftime('%Y-%m-%d')
    data_pd = fetch_base(begin_date, end_date)
    factors_data = factors_data.set_index(['trade_date','code'])
    factors_columns = factors_data.columns
    chg_pct = data_pd['ret_f1r_cc']
    dummy120_fst = data_pd['dummy120_fst']
    dummy120_fst_close = data_pd['dummy120_fst_close']
    hs300 = data_pd['hs300']
    #zz500 = data_pd['zz500']
    #zz1000 = data_pd['zz1000']
    res1 = []
    pdb.set_trace()
    for ff in factors_columns:
        print(ff)
        factor_data = factors_data[ff]
        factor_data = factor_data.unstack()
        factor_data = factor_data.reindex(dummy120_fst.index,
                                      columns=dummy120_fst.columns)

        #factor_data = factor_data[
        #    (hs300 == 1) | (zz500 == 1) |
        #    (zz1000 == 1)] * dummy120_fst_close * dummy120_fst
        factor_data = factor_data * dummy120_fst_close * dummy120_fst
        factor_data = alk_standardize(alk_winsorize(factor_data))
        factor_data = factor_data.stack()
        factor_data.name = ff
        factor_data = factor_data.sort_index()
        res1.append(factor_data)
    total_data = pd.concat(res1,axis=1)
    total_data = total_data.unstack().fillna(method='ffill').fillna(0)
    total_data = total_data.stack()
    filename = os.path.join(dirs, "normal_factors.feather")
    total_data.reset_index().to_feather(filename)

def build_data(method=os.environ['DUMMY_NAME'], horizon=1):
    pdb.set_trace()
    dirs = os.path.join(os.environ['BASE_PATH'], method, 'evolution', str(horizon))
    filename = os.path.join(dirs, "normal_factors.feather")
    normal_factors_data = pd.read_feather(filename)

    filename = os.path.join(os.environ['BASE_PATH'], method,
                            "normal_yields.feather")
    normal_yields_data = pd.read_feather(filename)

    total_data = normal_factors_data.merge(normal_yields_data, on=['trade_date', 'code'])
    total_data = total_data[(total_data['trade_date'] > '2020-04-04')&(total_data['trade_date'] < '2024-09-10')]
    total_data = total_data.sort_values(by=['trade_date','code'])
    dates = total_data['trade_date'].dt.strftime('%Y-%m-%d').unique().tolist()

    pos = int(len(dates) * 0.7)
    train_data = total_data[total_data['trade_date'].isin(dates[:pos])]
    val_data = total_data[total_data['trade_date'].isin(dates[pos:])]
    pdb.set_trace()
    ##切割样本
    train_filename = os.path.join(
        os.environ['BASE_PATH'], method,'evolution', str(horizon),
        "train_model_normal.feather")
    train_data.reset_index(drop=True).to_feather(train_filename)

    val_filename = os.path.join(os.environ['BASE_PATH'], method,'evolution', str(horizon),
                                "val_model_normal.feather")
    val_data.reset_index(drop=True).to_feather(val_filename)




#normal_factors(method=os.environ['DUMMY_NAME'], horizon=5)
build_data(method=os.environ['DUMMY_NAME'], horizon=5)
