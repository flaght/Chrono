import pandas as pd
import os, pdb, itertools

from ultron.ump.core.process import add_process_env_sig
from ultron.factor.genetic.geneticist.operators import calc_factor
from kdutils.process import split_k, create_parellel, run_process
from kdutils.data import fetch_d1factors, fetch_chgpct
from kdutils.data import fetch_base
#from kdutils.sqta import fetch_factors
from kdutils.kdmetrics import long_metrics
from dotenv import load_dotenv

load_dotenv()

## 去极值
def normal_data(method):
    pdb.set_trace()
    dfactors_data = fetch_d1factors(base_path=os.environ['FACTOR_PATH'])
    dirs = os.path.join(os.environ['BASE_PATH'], method)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    pdb.set_trace()
    filename = os.path.join(dirs, "winsorize_factors.feather")
    dfactors_data.reset_index().to_feather(filename)

## 加载未标准化的收益率
def build_data(method, horizon):
    pdb.set_trace()
    filename = os.path.join(os.environ['BASE_PATH'], method,
                            "{0}h_yields.feather".format(horizon))
    
    horizon_data = pd.read_feather(filename)

    filename = os.path.join(os.environ['BASE_PATH'], method, "winsorize_factors.feather")
    dfactors_data = pd.read_feather(filename)
    total_data = dfactors_data.merge(horizon_data, on=['trade_date','code'])
    total_data.rename(columns={'nxt1_ret':'nxt1_ret'},inplace=True)
    filename = os.path.join(os.environ['BASE_PATH'], method, "evolution_factors_{0}.feather".format(horizon))
    pdb.set_trace()
    total_data.reset_index(drop=True).to_feather(filename)

def metrics_factors(method, horizon):
    filename = os.path.join(os.environ['BASE_PATH'], method, "evolution_factors.feather")
    factors_data = pd.read_feather(filename)
    factors_data = factors_data.sort_values(by=['trade_date','code']).set_index('trade_date')
    ## 计算因子
    #formual = "MACCBands(14,'singlevolpropentropymin',MMIN(10,RSI(10,'mmindiffodaddprcstd')))"
    #formual = "MACCBands(14,MDPO(12,MACCBands(14,MKURT(12,MMAX(6,MVARIANCE(12,MARGMIN(20,'mcorrtotbidnandactselln')))),'simple_dis_prop_min')),MACCBands(14,MACCBands(14,MACCBands(14,'mcorrtotbidnandstdactsellv',MMAX(16,EMA(20,'mmindiffodaddprcstd'))),SIGLOG10ABS(DELTA(4,MRSquared(6,DELTA(4,'singlevolpropentropymin'),'dayvolatilityratiomin')))),RSI(10,'dayvolatilityratiomin')))"
    #formual = "MIChimoku(10,MSTD(20,MARGMAX(20,MARGMAX(20,'corr_prevret_adjamount_min'))),MMIN(20,'foundersurgevars5')) "
    formual = "WMA(4,CSRank(MADecay(8,MSUM(20,'foundersurgevars5w30'))))"
    factors_data = calc_factor(expression=formual,
                               total_data=factors_data,
                               key='code',
                               name='ultron_1734008271850764',
                               indexs=[])
    factors_data = factors_data.reset_index()
    factors_data['trade_date'] = pd.to_datetime(factors_data['trade_date'])

    begin_date = factors_data['trade_date'].min()
    end_date = factors_data['trade_date'].max()
    factors_data0 = factors_data.set_index(['trade_date','code'])['ultron_1734008271850764'].unstack() 
    remain_data = fetch_base(begin_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d'))
    ret_data = remain_data['ret_f1r_cc']
    dummy120_fst = remain_data['dummy120_fst']
    dummy120_fst_close = remain_data['dummy120_fst_close']
    hs300 = remain_data['hs300']
    zz1000 = remain_data['zz1000']
    zz500 = remain_data['zz500']

    pdb.set_trace()
    yields_data = ret_data.reindex(dummy120_fst.index, columns=dummy120_fst.columns)
    yields_data = yields_data[(hs300 == 1) | (zz500 == 1) | (zz1000 == 1)] * dummy120_fst_close * dummy120_fst
    #yields_data = yields_data[(hs300 == 1)] * dummy120_fst_close * dummy120_fst
    dummy_fst = dummy120_fst_close * dummy120_fst


    factors_data0 = factors_data0.reindex(dummy120_fst.index, columns=dummy120_fst.columns)
    factors_data0 = factors_data0[(hs300 == 1) | (zz500 == 1) | (zz1000 == 1)] * dummy_fst

    yields_data0 = yields_data.reindex(factors_data0.index,
                                           columns=factors_data0.columns)
    dummy_fst0 = dummy_fst.reindex(factors_data0.index,
                                       columns=factors_data0.columns)
        
    st0 = long_metrics(dummy_fst=dummy_fst0, yields_data=yields_data0,
                    factor_data=factors_data0, name='cols')

    print(st0)

def run_factors(column, basic_factors):
    print(column)
    factors_data = calc_factor(expression=column['formual'],
                               total_data=basic_factors,
                               key='code',
                               name=column['name'],
                               indexs=[])
    factors_data =  factors_data.reset_index().sort_values(
        ['trade_date', 'code']).set_index(['trade_date', 'code'])
    return factors_data



@add_process_env_sig
def create_factors(target_column, basic_factors):
    factors_data = run_process(target_column=target_column,
                       callback=run_factors,
                       basic_factors=basic_factors)
    #print(factors_data)
    return factors_data


def build_factors(method, horizon):
    ### 加载因子表达式
    pdb.set_trace()
    filename =  os.path.join("formual", "sqta_{0}.csv".format(horizon))
    formual_data = pd.read_csv(filename, index_col=0)
    ### 加载数据
    pdb.set_trace()
    filename = os.path.join(os.environ['BASE_PATH'], method, "winsorize_factors.feather")
    basic_factors = pd.read_feather(filename)
    basic_factors = basic_factors.sort_values(by=['trade_date','code']).set_index('trade_date')
    target_columns = formual_data[['name', 'formual']].to_dict('records')

    process_list = split_k(64, target_columns)

    res = create_parellel(process_list=process_list,
                          callback=create_factors,
                          basic_factors=basic_factors)
    pdb.set_trace()
    res = list(itertools.chain.from_iterable(res))
    factors_data = pd.concat(res,axis=1)
    dirs = os.path.join(os.environ['BASE_PATH'], method, 'evolution', str(horizon))
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    filename = os.path.join(dirs, "factors_data.feather")
    factors_data.reset_index().to_feather(filename)


    
def main(method):
    #normal_data(method)
    build_data(method, horizon=1)
    #build_factors(method=method, horizon=1)

main(method=os.environ['DUMMY_NAME'])