import sys, os, torch, pdb, argparse, re, datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

from alphakit.portfolio import *
from ultron.strategy.models.processing import winsorize as alk_winsorize
from ultron.strategy.models.processing import standardize as alk_standardize
from kdutils.data import fetch_basic, fetch_basic1, fetch_benckmark
bpath = os.path.join("./temp/portfolio")

def fetch_values(file_path, file_name, model_types, factor_name):
    dirs = os.path.join(os.environ['BASE_PATH'], os.environ['DUMMY_NAME'], model_types, 
            file_path)
    pdb.set_trace()
    filename = os.path.join(dirs, "{0}.feather".format(file_name))
    data = pd.read_feather(filename).rename(columns={'trade_time':'trade_date'})
    return data[['trade_date','code', factor_name]].set_index(['trade_date','code']).unstack()[factor_name]

def ToNWeight(file_path, file_name, model_types, factor_name, horizon):
    factors = fetch_values(file_path=file_path.format(horizon), file_name=file_name, model_types=model_types,
            factor_name=factor_name)
    begin_date = factors.index.min()# - datetime.timedelta(days=20)#.strftime('%Y-%m-%d')
    end_date = factors.index.max()# + datetime.timedelta(days=20)
    val, ret, iret, ret_c2o, usedummy,vardummy = fetch_basic(
        begin_date=begin_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'))
    weight = TopNWeight(vardummy, factors, 1, 200, 0)

    out1, pnl1, tvs1 = CalRet(usedummy, weight, ret, None, iret['000300'], 252)
    out2, pnl2, tvs2 = CalRet(usedummy, weight, ret, ret_c2o, iret['000300'], 252)

    dirs = os.path.join(bpath, model_types, str(horizon), 'top')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    weight.reset_index().to_feather(os.path.join(dirs, "weight.fea"))
    pnl1.name = 'pnl'
    pnl1.reset_index().to_feather(os.path.join(dirs, "pnl1.fea"))
    out1.index.name = 'name'
    out1.name = 'value'
    out1.reset_index().to_feather(os.path.join(dirs, "out1.fea"))

    pnl2.name = 'pnl'
    pnl2.reset_index().to_feather(os.path.join(dirs, "pnl2.fea"))
    out2.index.name = 'name'
    out2.name = 'value'
    out2.reset_index().to_feather(os.path.join(dirs, "out2.fea"))



def optimize1(file_path, file_name, model_types, factor_name, horizon):
    factors = fetch_values(file_path=file_path.format(horizon), file_name=file_name, model_types=model_types,
            factor_name=factor_name)
    benchmark = 'hs300wt'
    begin_date = factors.index.min()#.strftime('%Y-%m-%d')
    end_date = factors.index.max()# + datetime.timedelta(days=20)
    val1, ret1, iret1, ret_c2o1, usedummy1, vardummy1 = fetch_basic(
        begin_date=begin_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'))

    risk_exposure, risk_cov, specific_risk, weighted = fetch_benckmark(
            begin_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), benchmark)
    dummy, useweight, val = fetch_basic1(begin_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    factors = factors.reindex(index=dummy.index,columns=dummy.columns)
    er = alk_standardize(alk_winsorize(factors * dummy)) * useweight
    configure = {
            'name': 'test',
            'turn_over_target': 1,
            'risk_ub': 10,
            'risk_penalty': 0,
            'cost_penalty': 0,
            'buy_cost': 0.0015,
            'sell_cost': 0.0015,
            'activew_lower': -0.002,
            'activew_upper': 0.002,
            'lbound': 0.0,
            'ubound': 0.20,
            'liqu_ub': 1,
            'benchmark_lower': 0.99,
            'benchmark_upper': 1.01,
            'total_lower': 0.99,
            'total_upper': 1.01,
            'industry_effective': BARRA_INDUSTRY,
            'effective_industry_lower': -0.01,
            'effective_industry_upper': 0.01,
            'riskstyle': BARRA_RISKFACTOR,
            'riskstyle_lower': -0.2}

    positions = mosek_opt(er, configure, dummy, weighted, val, risk_exposure, risk_cov, specific_risk, begin_date, end_date, None)
    weight = positions.set_index(['trade_date','code']).unstack()

    pdb.set_trace()
    out1, pnl1, tvs1 = CalRet(usedummy1, weight, ret1, None, iret1['000300'], 252)
    out2, pnl2, tvs2 = CalRet(usedummy1, weight, ret1, ret_c2o1, iret1['000300'], 252)
    dirs = os.path.join(bpath, model_types, str(horizon), 'opt')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    weight['wt'].reset_index().to_feather(os.path.join(dirs, "weight.fea"))


    pnl1.name = 'pnl'
    pnl1.reset_index().to_feather(os.path.join(dirs, "pnl1.fea"))
    out1.index.name = 'name'
    out1.name = 'value'
    out1.reset_index().to_feather(os.path.join(dirs, "out1.fea"))

    pnl2.name = 'pnl'
    pnl2.reset_index().to_feather(os.path.join(dirs, "pnl2.fea"))
    out2.index.name = 'name'
    out2.name = 'value'
    out2.reset_index().to_feather(os.path.join(dirs, "out2.fea"))

            

    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='envoy0003_transient_hybrid_transformer_3p_2s_ranking_{0}h_dubhe_hs300_10c')
    parser.add_argument('--file_name', type=str, default='envoy0003_factors')
    parser.add_argument('--model_types', type=str, default='Timing')
    parser.add_argument('--factor_name', type=str, default='factor')
    parser.add_argument('--horizon', type=int, default=1)

    args = parser.parse_args()
    args = vars(args)
    ToNWeight(file_path=args['file_path'], file_name=args['file_name'],
            model_types=args['model_types'], factor_name=args['factor_name'],
            horizon=args['horizon'])
    optimize1(file_path=args['file_path'], file_name=args['file_name'],
                model_types=args['model_types'], factor_name=args['factor_name'],
                horizon=args['horizon'])