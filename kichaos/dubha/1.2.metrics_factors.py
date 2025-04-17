import os, pdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from kdutils.data import fetch_base
from kdutils.kdmetrics import long_metrics

def fetch_data():
    begin_date = None
    end_date = None
    res = {}
    base_path=os.environ['FACTOR_PATH']
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

    remain_data = fetch_base(begin_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d'))
    ret_data = remain_data['ret_f1r_cc']
    dummy120_fst = remain_data['dummy120_fst']
    dummy120_fst_close = remain_data['dummy120_fst_close']
    hs300 = remain_data['hs300']
    zz1000 = remain_data['zz1000']
    zz500 = remain_data['zz500']

    yields_data = ret_data.reindex(dummy120_fst.index, columns=dummy120_fst.columns)
    #yields_data = yields_data[(hs300 == 1) | (zz500 == 1) | (zz1000 == 1)] * dummy120_fst_close * dummy120_fst
    yields_data = yields_data[(hs300 == 1)] * dummy120_fst_close * dummy120_fst
    dummy_fst = dummy120_fst_close * dummy120_fst
    
    pdb.set_trace()
    res1=[]
    for col in res:
        print(col)
        factors_data0 = res[col].copy()
        factors_data0 = factors_data0.reindex(dummy120_fst.index, columns=dummy120_fst.columns)
        #factors_data0 = factors_data0[(hs300 == 1) | (zz500 == 1) | (zz1000 == 1)] * dummy_fst
        factors_data0 = factors_data0[(hs300 == 1)] * dummy_fst

        yields_data0 = yields_data.reindex(factors_data0.index,
                                           columns=factors_data0.columns)
        dummy_fst0 = dummy_fst.reindex(factors_data0.index,
                                       columns=factors_data0.columns)
        
        st0 = long_metrics(dummy_fst=dummy_fst0, yields_data=yields_data0,
                    factor_data=factors_data0, name=col)
        res1.append(st0)
    pdb.set_trace()
    print('-->')
fetch_data()