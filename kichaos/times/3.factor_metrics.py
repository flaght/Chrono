import os, pdb, inspect, re, time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from ultron.factor.dimension import DimensionCorrcoef
from ultron.factor.dimension.corrcoef import FCorrType
from ultron.ump.similar.corrcoef import ECoreCorrType
from ultron.ump.similar.corrcoef import corr_xy, corr_matrix, ECoreCorrType

import lumina.techinical as lt
import lumina.features as lf

from lumina.features.feature_adosc import FeatureAdosc as Feature
from lumina.features.feature_annealn import FeatureAnnealn as Feature
from lumina.features.feature_ao import FeatureAO as Feature
from lumina.features.feature_aobv import FeatureAobv as Feature
from lumina.features.feature_apo import FeatureAPO as Feature
from lumina.features.feature_bop import FeatureBOP as Feature
from lumina.features.feature_brar import FeatureBRAR as Feature
from lumina.features.feature_cci import FeatureCCI as Feature
from lumina.features.feature_chkbar import FeatureCHKBAR as Feature
from lumina.features.feature_clkbar import FeatureCLKBAR as Feature
from lumina.features.feature_cmf import FeatureCMF as Feature
from lumina.features.feature_dema import FeatureDema as Feature
from lumina.features.feature_efi import FeatureEFI as Feature
from lumina.features.feature_ichimoku import FeatureIchimoku as Feature
from lumina.features.feature_kvo import FeatureKVO as Feature
from lumina.features.feature_vwma  import FeatureVWMA as Feature

pd.options.mode.copy_on_write = True
load_dotenv()

# getattr(lt, "calc_{0}".format(name.lower()))(data.loc[:28317], *params).tl
def factor_metrics(method, scaler_name, horizon, offset=1):
    file_name = os.path.join(
        os.getenv('BASE_PATH'), 'times', os.environ['CODE'],
        f'{method}_lumina_{scaler_name}_{horizon}_{offset}_factors.feather')
    total_data = pd.read_feather(file_name)
    total_data = total_data.dropna(subset=['nxt1_ret'])
    features = [
        col for col in total_data.columns
        if col not in ['trade_time', 'code', 'price', 'chg', 'nxt1_ret']
    ]
    res = {}
    for f in features:
        print(f)
        s1 = corr_xy(total_data[f], total_data['nxt1_ret'],
                     ECoreCorrType.E_CORE_TYPE_SPERM)
        res[f] = s1


def load_data(method, code):
    file_path = os.path.join(os.getenv('BASE_PATH'), 'times', code,
                             f'{method}.feather')
    print(f'Loading {file_path}')
    data = pd.read_feather(file_path)
    return data.reset_index(drop=True)


## 收益率处理
def create_yields(data, horizon, offset=1):
    dt = data.set_index(['trade_time'])
    dt["nxt1_ret"] = dt['chg']
    dt = dt.groupby("code").rolling(
        window=horizon, min_periods=1)['nxt1_ret'].sum().groupby(level=0)
    dt = dt.shift(0).unstack().T.shift(-(horizon + offset - 1)).stack(
        dropna=False)
    dt.name = 'nxt1_ret'
    data = dt.reset_index().merge(data, on=['trade_time',
                                            'code']).reset_index(drop=True)
    return data.dropna(subset=['nxt1_ret'])


def calc_factors(data):
    indexs = data.set_index(['trade_time', 'code']).index
    count, _ = data.shape
    res = []
    keys_vars = {
        attr: value
        for attr, value in Feature().__dict__.items() if attr.endswith('_keys')
    }
    key_name = list(keys_vars.keys())[0]
    keys = keys_vars[key_name]
    name = key_name.split('_keys')[0]
    i = 0
    for k in keys:
        i += 1
        params = k if isinstance(k, tuple) else (k, )
        ny = getattr(lt, "calc_{0}".format(name.lower()))(data, *params)
        
        if isinstance(ny, tuple):
            for j in range(len(ny)):
                fname = "{0}_{1}_{2}".format(name, i, j)
                dt = pd.Series(ny[j].tl, index=indexs, name=fname)
                assert (dt.dropna().shape[0] == count)
                res.append(dt)
        else:
            fname = "{0}_{1}".format(name, i)
            dt = pd.Series(ny.tl, index=indexs, name=fname)
            assert (dt.dropna().shape[0] == count)
            res.append(dt)
    return pd.concat(res, axis=1).reset_index()


def test_factors(method, codes, horizon, offset=1):
    res = []
    for code in codes:
        data = load_data(method=method, code=code).dropna(subset=['chg'])
        data = data.sort_values(['trade_time', 'code']).reset_index(drop=True)
        factors_data = calc_factors(data)
        res.append(factors_data)
    ## 收益率计算
    factors_data = pd.concat(res, axis=0)
    factors_data = factors_data.merge(data[['trade_time', 'code', 'chg']],
                                      on=['trade_time', 'code'])
    factors_data = factors_data.sort_values(['trade_time', 'code'])
    factors_data = create_yields(factors_data, int(horizon))

    features = [
        col for col in factors_data.columns
        if col not in ['trade_time', 'code', 'nxt1_ret', 'chg']
    ]
    res = {}
    for f in features:
        print(f)
        s1 = corr_xy(factors_data[f], factors_data['nxt1_ret'],
                     ECoreCorrType.E_CORE_TYPE_SPERM)
        res[f] = s1
    
    print(res)



codes = ['IM', 'IH', 'IC', 'IF']
test_factors(method=os.getenv('METHOD'),
             codes=codes,
             horizon=os.getenv('HORIZON'),
             offset=1)
#metrics(os.getenv('METHOD'), os.getenv('SCALER'), int(os.getenv('HORIZON')))
