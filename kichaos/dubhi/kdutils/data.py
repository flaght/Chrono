import os, pdb
import pandas as pd

### 映射
from alphakit.const import *
from alphakit.data import *
from alphacopilot.api.data import RetrievalAPI, ddb_tools, DDBAPI
from ultron.strategy.models.processing import winsorize as alk_winsorize
from ultron.strategy.models.processing import standardize as alk_standardize
from ultron.tradingday import advanceDateByCalendar
#from ultron.factor.metrics.CrossSection.booster import Booster

def fetch_base(begin_date, end_date):
    data = RetrievalAPI.get_data_by_map(
        columns=['ret_f1r_cc', 'dummy120_fst', 'dummy120_fst_close','hs300', 'zz1000', 'zz500'],
        begin_date=begin_date,
        end_date=end_date,
        method='ddb',
        is_debug=True)
    
    return data


def fetch_chgpct(begin_date,end_date):
    #index = ['ret_f1r_cc', 'dummy120_fst', 'dummy120_fst_close','hs300', 'zz1000', 'zz500']
    #index = ['ret_f1r_cc', 'dummy120_fst', 'dummy120_fst_close','hs300']
    index = ['ret_f1r_cc', 'dummy120_fst', 'dummy120_fst_close']
    data = RetrievalAPI.get_data_by_map(
        columns=index,
        begin_date=begin_date,
        end_date=end_date,
        method='ddb',
        is_debug=True)
    
    chg_pct = data['ret_f1r_cc']
    dummy120_fst = data['dummy120_fst']
    dummy120_fst_close = data['dummy120_fst_close']
    #hs300 = data['hs300']
    #zz500 = data['zz500']
    #zz1000 = data['zz1000']
    #chg_pct = chg_pct.reindex(dummy120_fst.index, columns=dummy120_fst.columns)
    #chg_pct = chg_pct[
    #        (hs300 == 1) | (zz500 == 1) |
    #        (zz1000 == 1)] * dummy120_fst_close * dummy120_fst
    #chg_pct = chg_pct[(hs300==1)] * dummy120_fst_close * dummy120_fst
    #chg_pct = chg_pct * dummy120_fst_close * dummy120_fst
    #chg_pct = alk_standardize(alk_winsorize(chg_pct))
    chg_pct = chg_pct.stack()
    #chg_pct = chg_pct.unstack()#.fillna(0)
    #chg_pct = chg_pct.stack()
    chg_pct.name = 'chg_pct'
    return chg_pct.reset_index()


def fetch_d1factors(base_path):
    res = {}
    res1 = []
    pdb.set_trace()
    begin_date = None
    end_date = None
    pdb.set_trace()
    #booster = Booster(hold=1, skip=0)
    filter_cols = ['flinfr_c0', 'flinfr_cstd', 'peak', 'consist_ab_vol_min', 'morningfogmin']
    for root, dirs, files in os.walk(base_path):
        for filename in files:
            name = filename.split('.')[0]
            if name in filter_cols:
                continue
            factor = pd.read_feather(os.path.join(base_path, filename))
            min_date = factor['trade_date'].min()
            max_date = factor['trade_date'].max()
            #res.append({'name':name, 'min_date':min_date.strftime('%Y-%m-%d'), 'max_date':max_date.strftime('%Y-%m-%d')})
            begin_date = min_date if begin_date is None or begin_date < min_date else begin_date
            end_date = max_date if end_date is None or end_date >  max_date else end_date
            factor = factor.set_index('trade_date')
            columns = factor.columns
            columns = [col.zfill(6) for col in columns]
            factor.columns = columns
            factor.name = name
            res[name] = factor
    #columns = ['sw1', 'dummy120_fst', 'dummy120_fst_close', 
    #            'hs300', 'zz1000', 'zz500']
    #columns = ['sw1', 'dummy120_fst', 'dummy120_fst_close','hs300']
    columns = ['sw1', 'dummy120_fst', 'dummy120_fst_close']
    data_pd = RetrievalAPI.get_data_by_map(
        columns=columns,
        begin_date=begin_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        method='ddb',
        is_debug=True)
    
    ### 标准化
    filter1 = []
    total_data = None
    dummy120_fst = data_pd['dummy120_fst']
    dummy120_fst_close = data_pd['dummy120_fst_close']
    #hs300 = data_pd['hs300']
    #zz1000 = data_pd['zz1000']
    #zz500 = data_pd['zz500']
    res1 = []
    for ff in res:
        print(ff)
        factor_data = res[ff]
        factor_data = factor_data.reindex(dummy120_fst.index,
                                      columns=dummy120_fst.columns)

        #factor_data = factor_data[
        #    (hs300 == 1) | (zz500 == 1) |
        #    (zz1000 == 1)] * dummy120_fst_close * dummy120_fst
        #factor_data = factor_data[(hs300 == 1)] * dummy120_fst_close * dummy120_fst
        #factor_data = factor_data * dummy120_fst_close * dummy120_fst
        #factor_data = alk_winsorize(factor_data)
        factor_data = factor_data.stack()
        factor_data.name = ff
        factor_data = factor_data.sort_index()
        res1.append(factor_data)
        #factor_data = factor_data.sort_index().reset_index()
        #total_data = factor_data if total_data is None else total_data.merge(factor_data, on=['trade_date','code'])
    total_data = pd.concat(res1,axis=1)
    total_data = total_data.unstack().fillna(method='ffill').fillna(0)
    total_data = total_data.stack()
    return total_data

def fetch_dfactors(base_path):
    res = {}
    res1 = []
    begin_date = None
    end_date = None
    #booster = Booster(hold=1, skip=0)
    pdb.set_trace()
    filter_cols = ['flinfr_c0', 'flinfr_cstd', 'peak', 'consist_ab_vol_min', 'morningfogmin']
    for root, dirs, files in os.walk(base_path):
        for filename in files:
            name = filename.split('.')[0]
            if name in filter_cols:
                continue
            factor = pd.read_feather(os.path.join(base_path, filename))
            min_date = factor['trade_date'].min()
            max_date = factor['trade_date'].max()
            #res.append({'name':name, 'min_date':min_date.strftime('%Y-%m-%d'), 'max_date':max_date.strftime('%Y-%m-%d')})
            begin_date = min_date if begin_date is None or begin_date < min_date else begin_date
            end_date = max_date if end_date is None or end_date >  max_date else end_date
            factor = factor.set_index('trade_date')
            columns = factor.columns
            columns = [col.zfill(6) for col in columns]
            factor.columns = columns
            factor.name = name
            res[name] = factor
    #columns = ['sw1', 'dummy120_fst', 'dummy120_fst_close', 
    #            'hs300', 'zz1000', 'zz500']
   # columns = ['sw1', 'dummy120_fst', 'dummy120_fst_close','hs300']
    pdb.set_trace()
    columns = ['sw1', 'dummy120_fst', 'dummy120_fst_close']
    data_pd = RetrievalAPI.get_data_by_map(
        columns=columns,
        begin_date=begin_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        method='ddb',
        is_debug=True)
    
    ### 标准化
    filter1 = []
    total_data = None
    dummy120_fst = data_pd['dummy120_fst']
    dummy120_fst_close = data_pd['dummy120_fst_close']
    #hs300 = data_pd['hs300']
    #zz1000 = data_pd['zz1000']
    #zz500 = data_pd['zz500']
    res1 = []
    for ff in res:
        print(ff)
        factor_data = res[ff]
        #factor_data = factor_data.reindex(dummy120_fst.index,
        #                              columns=dummy120_fst.columns)
        
        #factor_data = factor_data * dummy120_fst_close * dummy120_fst
        #factor_data = alk_standardize(alk_winsorize(factor_data))
        #factor_data = factor_data.stack()
        #factor_data.name = ff
        factor_data = factor_data.sort_index()
        factor_data = factor_data.stack()
        factor_data = factor_data.sort_index()
        factor_data.name = ff
        res1.append(factor_data)
        #factor_data = factor_data.sort_index().reset_index()
        #total_data = factor_data if total_data is None else total_data.merge(factor_data, on=['trade_date','code'])
    pdb.set_trace()
    total_data = pd.concat(res1,axis=1)
    #total_data = total_data.unstack().fillna(method='ffill').fillna(0)
    #total_data = total_data.stack()
    return total_data


def fetch_basic(begin_date, end_date):
    data = RetrievalAPI.get_data_by_map(columns=[
        'ret_o2o', 'turnoverValue', 'dummy_test_f1r_open', 'dummy120_fst_close',
        'ret', 'iret', 'ret_c2o', 'zz1000', 'hs300', 'dummy120_fst'
        ],
                                    begin_date=begin_date,
                                    end_date=end_date,
                                    method='ddb',
                                    is_debug=True)
    val = data['turnoverValue']
    ret = data['ret']
    iret = data['iret']
    ret_c2o = data['ret_c2o']
    usedummy = data['dummy_test_f1r_open'] * data['dummy120_fst_close']
    vardummy = data['dummy120_fst_close']
    return val, ret, iret, ret_c2o, usedummy,vardummy

def fetch_basic1(begin_date, end_date):
    start_date = advanceDateByCalendar(
        'china.sse', begin_date, '-{0}b'.format(120)).strftime('%Y-%m-%d')
    data = RetrievalAPI.get_data_by_map(
        columns=['dummy120_fst_close', 'negMarketValue', 'turnoverValue'],
        begin_date=start_date,
        end_date=end_date,
        method='ddb')
    dummy = data['dummy120_fst_close']
    fcap = data['negMarketValue']
    val = data['turnoverValue']
    val[val.isna()] = 0
    val = val.rolling(20).sum() / 20

    retstd = 1 / np.sqrt(fcap.rolling(window=120, min_periods=1).mean())
    return dummy.loc[begin_date:], retstd.loc[begin_date:], val.loc[
        begin_date:]

def fetch_benckmark(begin_date, end_date, benchmark):
    start_date = advanceDateByCalendar('china.sse', begin_date,
                                       '-{0}b'.format(21)).strftime('%Y-%m-%d')
    data = RetrievalAPI.get_data_by_map(columns=BARRA_ALL_K,
                                        begin_date=begin_date,
                                        end_date=end_date,
                                        method='ddb',
                                    is_debug=True)
    risk_exposure = getdataset(data, BARRA_ALL_K)
    risk_exposure.columns = risk_exposure.columns.str.removeprefix('f_')
    risk_exposure.reset_index(inplace=True)

    covstr = list('fcov_' + i for i in BARRA_ALL)
    data = RetrievalAPI.get_data_by_map(columns=covstr,
                                        begin_date=begin_date,
                                        end_date=end_date,
                                        method='ddb',
                                    is_debug=True)
    risk_cov = pd.DataFrame()
    for f in covstr:
        factor = data[f].unstack()
        factor.name = f
        risk_cov = pd.concat([risk_cov, factor], axis=1)
    risk_cov.columns = risk_cov.columns.str.removeprefix('fcov_')
    risk_cov.reset_index(inplace=True)
    risk_cov.rename(columns={
        'level_0': 'Factor',
        'level_1': 'trade_date'
    },
                    inplace=True)
    pdb.set_trace()
    data = RetrievalAPI.get_data_by_map(columns=['SRISK', 'sw1'] + [benchmark],
                                        begin_date=start_date,
                                        end_date=end_date,
                                        method='ddb',
                                    is_debug=True)
    specific_risk = data['SRISK']
    weighted = data[benchmark].rolling(20, min_periods=1).sum() / 20

    return risk_exposure.loc[
        risk_exposure.trade_date >= begin_date], risk_cov.loc[
            risk_exposure.trade_date >= begin_date], specific_risk.loc[
                begin_date:], weighted.loc[begin_date:]
