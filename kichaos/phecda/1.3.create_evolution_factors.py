import pandas as pd
import os, pdb, math, itertools
from dotenv import load_dotenv

load_dotenv()

from jdw import EntropyAPI
from kdutils.macro import base_path, codes
from kdutils.process import split_k, create_parellel, run_process, add_process_env_sig
from alphacopilot.api.calendars import advanceDateByCalendar
from ultron.factor.genetic.geneticist.operators import calc_factor
from ultron.factor.metrics.TimeSeries.metrics import Metrics

g_instruments = 'ims'  #os.environ['INSTRUMENTS']

# 'MRSquared',
two_operators_sets = [
    'MConVariance', 'MMASSI', 'MACCBands', 'MPWMA', 'MIChimoku', 'MRes',
    'MMeanRes', 'MCORR', 'MCoef', 'MSLMean', 'MSmart', 'MSharp', 'MSortino',
    'MINIMUM', 'MAXIMUM', 'ADDED', 'SUBBED', 'MUL', 'DIV', 'MOD'
]
#, 'MHMA', 'MDPO', 'MARETURNLog',
one_operators_sets = [
    'MA', 'MPERCENT', 'MMedian', 'MADiff', 'MADecay', 'MMAX', 'MMIN', 'MDPO',
    'MARGMAX', 'MARGMIN', 'MRANK', 'MQUANTILE', 'MCPS', 'MDIFF', 'MMaxDiff',
    'MMinDiff', 'MSUM', 'MPRO', 'MVARIANCE', 'MVHF', 'MDPO', 'MT3', 'MDEMA',
    'MIR', 'MSKEW', 'MKURT', 'MSTD', 'MNPOSITIVE', 'MAPOSITIVE', 'EMA', 'RSI',
    'WMA', 'MMaxDrawdown', 'MMDrawdown', 'SIGN', 'AVG', 'SQRT', 'DIFF', 'LOG2',
    'LOG10', 'LOG', 'EXP', 'FRAC', 'SIGLOG2ABS', 'SIGLOG10ABS', 'SIGLOGABS',
    'POW', 'ABS', 'ACOS', 'ASIN', 'NORMINV', 'CEIL', 'FLOOR', 'ROUND', 'TANH',
    'RELU', 'SHIFT', 'DELTA', 'SIGMOID', 'RETURNSimple', 'RETURNLog'
]

operators_sets = two_operators_sets + one_operators_sets


def callback_models(gen, rootid, best_programs, custom_params):
    tournament_size = custom_params['tournament_size']
    standard_score = custom_params['standard_score']
    dethod = custom_params['dethod']
    method = custom_params['method']
    best_programs = [program.output() for program in best_programs]
    best_programs = pd.DataFrame(best_programs)
    dirs = os.path.join(base_path, dethod, method, g_instruments, 'evolution')
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    names = custom_params[rootid]
    filename = os.path.join(dirs, f'{names}.feather')
    if os.path.exists(filename):
        old_dt = pd.read_feather(filename)
        best_programs = pd.concat([old_dt, best_programs], axis=0)
    best_programs = best_programs.drop_duplicates(subset=['name'])
    final_programs = best_programs[best_programs['fitness'] > standard_score]
    if final_programs.shape[0] < tournament_size:
        best_programs = best_programs.sort_values('fitness', ascending=False)
        final_programs = best_programs.head(tournament_size)
    final_programs.sort_values(
        'fitness', ascending=False).reset_index(drop=True).to_feather(filename)
    print(final_programs)
    print(filename)


def callback_fitness(factor_data, total_data, factor_sets, custom_params,
                     default_value):
    returns = total_data[['trade_time', 'code', 'nxt1_ret']]
    factor_data = factor_data.reset_index()
    data = factor_data.merge(returns, on=['trade_time', 'code'])
    data = data.set_index(['trade_time', 'code']).dropna(subset=['nxt1_ret'])
    factors_socre = data['transformed'].unstack()
    yield_score = data['nxt1_ret'].unstack()
    dummy = yield_score.copy()
    dummy.loc[:, :] = 1
    horizon = 1 if 'horizon' not in custom_params else custom_params['horizon']
    metrics = Metrics(dummy=dummy,
                      returns=yield_score,
                      factors=factors_socre,
                      hold=horizon,
                      skip=0,
                      group=10)
    results = metrics.fit()
    fitness = results.ic
    return math.fabs(fitness), results


def fetch_data(method, names):
    dirs = os.path.join(base_path, method, g_instruments, 'factors')
    res = []
    for filename in os.listdir(dirs):
        print(os.path.join(dirs, filename))
        dt = pd.read_feather(os.path.join(dirs, filename))
        dt = dt.set_index(['trade_time', 'code'])
        cols = [col for col in dt.columns if col in names]
        dt1 = dt[cols].sort_index() if len(cols) > 0 else dt
        if not dt1.empty:
            res.append(dt1)
    data = pd.concat(res, axis=1).sort_index()
    data = data.unstack().fillna(method='ffill')
    return data.stack().reset_index()


def fetch_returns(method, categories, horizon=1):
    dirs = os.path.join(base_path, method, g_instruments, 'yields')
    filename = os.path.join(dirs,
                            '{0}_{1}h.feather'.format(categories, horizon))
    return pd.read_feather(filename)


###### 因子挖掘
def evolution_factors(method, categories, horizon):
    rootid = 100001
    name = "{0}_{1}h".format(categories, horizon)
    configure = {
        'n_jobs': 16,
        'population_size': 50,
        'tournament_size': 20,
        'init_depth': 6,
        'evaluate': 'both_evaluate',
        'method': 'fitness',
        'crossover': 0.3,
        'point_replace': 0.2,
        'hoist_mutation': 0.2,
        'subtree_mutation': 0.2,
        'point_mutation': 0.1,
        'generations': 8,
        'standard_score': 0.0065,
        'rootid': rootid,
    }

    custom_params = {'horizon': horizon, rootid: name, 'dethod': method}
    factors_data = fetch_data(method, names=[])
    ret_f1r_oo = fetch_returns(method=method,
                               categories=categories,
                               horizon=horizon)
    pdb.set_trace()
    total_data = factors_data.merge(ret_f1r_oo, on=['trade_time', 'code'])
    begin_time = total_data['trade_time'].min()
    start_time = advanceDateByCalendar('china.sse', begin_time,
                                       '{0}b'.format(1)).strftime('%Y-%m-%d')
    total_data = total_data[total_data['trade_time'] >= start_time]
    total_data = total_data.set_index('trade_time')
    factor_columns = [
        col for col in factors_data.columns.tolist()
        if col not in ['trade_time', 'code']
    ]
    futures_engine = EntropyAPI.FuturesGeneticist(
        offset=0,
        horizon=horizon,
        factor_columns=factor_columns,
        universe=g_instruments,
        industry_name=g_instruments,
        industry_level=1,
        operators=operators_sets,
        is_loop=True,
        callback_fitness=callback_fitness,
        callback_save=callback_models)
    futures_engine.calculate_result(total_data=total_data,
                                    configure=configure,
                                    custom_params=custom_params)


#### 挖掘因子生成因子值
def run_factors(column, basic_factors):
    print(column)
    try:
        factors_data = calc_factor(expression=column['formual'],
                                   total_data=basic_factors,
                                   key='code',
                                   name=column['name'],
                                   indexs=[])
        factors_data = factors_data.reset_index().sort_values(
            ['trade_time', 'code']).set_index(['trade_time', 'code'])
    except Exception as e:
        print("error:{0}, column:{1}".format(e, column))
        factors_data = pd.DataFrame(index=['trade_time', 'code'])
    return factors_data


@add_process_env_sig
def create_factors(target_column, basic_factors):
    factors_data = run_process(target_column=target_column,
                               callback=run_factors,
                               basic_factors=basic_factors)
    #print(factors_data)
    return factors_data


def build_factors(method1='aicso1',
                  method2='kimto1',
                  categories='o2o',
                  horizon=1,
                  top=30):
    pdb.set_trace()
    if method1 == 'nicso':
        filter_cols = [
            'ultron_1731653779705408', 'ultron_1731588484951599',
            'ultron_1731674332172136', 'ultron_1731744349866204',
            'ultron_1731719589906514', 'ultron_1731768583509462',
            'ultron_1731652576323814', 'ultron_1731718809444019',
            'ultron_1731598152204620', 'ultron_1731612321481140',
            'ultron_1731608009499910', 'ultron_1731841989679380',
            'ultron_1731674507323525', 'ultron_1731750809117704',
            'ultron_1731651780656888', 'ultron_1731651955461328',
            'ultron_1731672683836960', 'ultron_1731817984755224',
            'ultron_1731652882367168', 'ultron_1731652884080889',
            'ultron_1731822733971052', 'ultron_1731552311260364',
            'ultron_1731750927316162', 'ultron_1731675242363532',
            'ultron_1731607884265850', 'ultron_1731748577684927',
            'ultron_1731802350957674', 'ultron_1731607071280834',
            'ultron_1731606504805954', 'ultron_1731608095971744',
            'ultron_1731815832705380', 'ultron_1731815081527757',
            'ultron_1731816236652928', 'ultron_1731589149901272',
            'ultron_1731653663310990', 'ultron_1731817378170410',
            'ultron_1731816915677812', 'ultron_1731817242221702',
            'ultron_1731820293737612', 'ultron_1731826992041996',
            'ultron_1731827014618494', 'ultron_1731824910127456',
            'ultron_1731826769395481', 'ultron_1731826358644296',
            'ultron_1731914137599279', 'ultron_1731827141353207',
            'ultron_1731651602855546', 'ultron_1731597045653020',
            'ultron_1731612499090414', 'ultron_1731746129057072',
            'ultron_1731750896871222', 'ultron_1731841403339402',
            'ultron_1731826751741880', 'ultron_1731826731071202',
            'ultron_1731824910125940', 'ultron_1731826815709473',
            'ultron_1731822493953588', 'ultron_1731820517497382',
            'ultron_1731605528291672', 'ultron_1731608295230856',
            'ultron_1731612343676012', 'ultron_1731795827606829',
            'ultron_1731603912922752', 'ultron_1731607922544674',
            'ultron_1731802237259636', 'ultron_1731812199474045',
            'ultron_1731821401494134', 'ultron_1731825451569896',
            'ultron_1731578868643278', 'ultron_1731579227620868',
            'ultron_1731732576204596', 'ultron_1731821317507561',
            'ultron_1731613467854042', 'ultron_1731924658094766',
            'ultron_1731814327127670', 'ultron_1731620576514250',
            'ultron_1731621943437682', 'ultron_1731819397045578',
            'ultron_1731821317440282', 'ultron_1731853484110502',
            'ultron_1731821318178170', 'ultron_1731812866947192',
            'ultron_1731924619576612', 'ultron_1731820424726850'
        ]

        filename = '/workspace/data/dev/kd/evolution/nn/phecda/fitness/rbb/evolution/20241113001.feather'
        formual_data = pd.read_feather(filename)
        formual_data = formual_data[formual_data['name'].isin(filter_cols)]
    else:
        ## 加载因子文件
        dirs = os.path.join(base_path, method1, 'fitness', g_instruments,
                            'evolution')
        filename = os.path.join(dirs,
                                '{0}_{1}h.feather'.format(categories, horizon))
        formual_data = pd.read_feather(filename)
        formual_data = formual_data[formual_data['fitness'] >= 0.028]  #0.01

    basic_factors = fetch_data(method2, names=[])
    begin_time = basic_factors['trade_time'].min()
    start_time = advanceDateByCalendar('china.sse', begin_time,
                                       '{0}b'.format(1)).strftime('%Y-%m-%d')
    basic_factors = basic_factors[basic_factors['trade_time'] >= start_time]
    basic_factors = basic_factors.sort_values(
        by=['trade_time', 'code']).set_index('trade_time')
    target_columns = formual_data[['name', 'formual']].to_dict('records')
    process_list = split_k(16, target_columns)

    res = create_parellel(process_list=process_list,
                          callback=create_factors,
                          basic_factors=basic_factors)
    res = list(itertools.chain.from_iterable(res))
    factors_data = pd.concat(res, axis=1)
    dirs = os.path.join(base_path, method2, 'evolution', g_instruments,
                        str(horizon))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    filename = os.path.join(dirs, '{0}_factors.feather'.format(categories))
    factors_data.reset_index().to_feather(filename)
    ### 同时保存使用的公式
    pdb.set_trace()
    filename = os.path.join(dirs, '{0}_formual.feather'.format(categories))
    formual_data.reset_index(drop=True).to_feather(filename)


if __name__ == '__main__':
    #evolution_factors(method='aicso2', categories='o2o', horizon=1)
    ### method1: 基于挖掘数据产生的因子表达式
    ### method2: 生成因子值
    build_factors(
        method1='aicso2',  #'nicso',
        method2='aicso3',
        categories='o2o',
        horizon=1,
        top=30)
