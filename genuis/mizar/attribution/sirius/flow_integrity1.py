### 因子值及绩效跟踪
import datetime, itertools
from collections import namedtuple
from dotenv import load_dotenv

load_dotenv()

from ultron.tradingday import *
from ultron.sentry.api import *
from lumina.formual.impulse import Impulse
from lumina.evolution.fusion.actuator import Actuator
from lib.uvx import load_sirius_params
from lib.cux003 import FactorEvaluate1
from lib.attr001.ftd001 import *


class FactorEvaluateTuple(
        namedtuple('FactorEvaluateTuple',
                   ('name', 'raw_factors', 'raw_returns', 'resample_data'))):
    __slots__ = ()


def clip_series_to_window(series_data, begin_time=None, end_time=None):
    if begin_time is None and end_time is None:
        return series_data

    clipped = series_data.copy()
    if not pd.api.types.is_datetime64_any_dtype(clipped.index):
        clipped.index = pd.to_datetime(clipped.index, errors="coerce")

    if begin_time is not None:
        clipped = clipped.loc[clipped.index >= pd.Timestamp(begin_time)]
    if end_time is not None:
        end_time1 = advanceDateByCalendar("china.sse", end_time, "1b")
        clipped = clipped.loc[clipped.index <= (pd.Timestamp(end_time1))]
    return clipped


def persist_evaluate_series(mongo_client,
                            eval_results,
                            category,
                            code,
                            begin_time=None,
                            end_time=None):

    for eval_result in eval_results:
        raw_factors = clip_series_to_window(eval_result.raw_factors,
                                            begin_time=begin_time,
                                            end_time=end_time)
        update_evaluate_series(mongo_client=mongo_client,
                               series_data=raw_factors,
                               table_name="realm_raw_factors",
                               factor_name=eval_result.name,
                               category=category,
                               code=code)

        df = raw_factors = clip_series_to_window(eval_result.resample_data,
                                                 begin_time=begin_time,
                                                 end_time=end_time)
        df = df.reset_index()
        df = df.drop([eval_result.name], axis=1)
        df['name'] = eval_result.name
        df['code'] = code
        df['category'] = category
        pdb.set_trace()
        update_netout_series2(mongo_client,
                              df_data=df,
                              table_name='realm_factors_metrics',
                              unique_keys=['trade_time', 'name', 'category'])

    update_returns_series(mongo_client=mongo_client,
                          series_data=eval_result._asdict()['raw_returns'],
                          table_name=f"realm_raw_returns",
                          code=code,
                          category=category)


def create_impulse(factors_infos, market_unstack):
    dependencies = [
        eval(formula['formula'])._dependency for formula in factors_infos
    ]
    dependencies = list(itertools.chain.from_iterable(dependencies))
    factors_data1 = Impulse(dependencies).batch(data=market_unstack)
    return factors_data1


def evaluate(factors_infos,
             normal_data,
             horizon,
             roll_win=15,
             fee=0.0,
             scale_method='raw'):
    res = []
    normal_data1 = normal_data.set_index(['trade_time', 'code'])
    for factor in factors_infos:
        factor_name = factor["formula"]
        evaluate1 = FactorEvaluate1(factor_data=normal_data,
                                    factor_name=factor_name,
                                    ret_name="nxt1_ret",
                                    roll_win=roll_win,
                                    fee=fee,
                                    scale_method=scale_method,
                                    expression=factor_name,
                                    resampling_win=horizon)
        _ = evaluate1.run()
        results = FactorEvaluateTuple(
            name=factor_name,
            raw_factors=normal_data1[factor_name].droplevel('code'),
            raw_returns=normal_data1['nxt1_ret'].droplevel('code'),
            resample_data=evaluate1.resample_data)
        res.append(results)
    return res


def run1(market_data, trading_sessions, factors_infos, params, begin_pos=32):
    market_data = filter_trading_time(data=market_data,
                                      trading_sessions=trading_sessions)
    market_data = market_data.set_index(['trade_time', 'code'])

    returns_data = create_returns(market_data=market_data,
                                  horizon=params['horizon'],
                                  name='close')

    market_unstack = market_data_format(market_data)

    ## 创建基础字段
    impulse_factors = create_impulse(factors_infos=factors_infos,
                                     market_unstack=market_unstack)

    ## 衍生计算， 标准化
    actuator = Actuator(k_split=1)
    original_factors, normal_factors = actuator.calculate(
        factors_infos=factors_infos,
        total_data=impulse_factors.reset_index(),
        method=params['method'],
        win=params['win'])

    ## 绩效计算
    normal_data = normal_factors.reset_index().merge(returns_data,
                                                     on=['trade_time', 'code'])

    normal_data = normal_data[begin_pos:normal_data.shape[0] -
                              params['horizon'] + 1]
    eval_data = evaluate(factors_infos=factors_infos,
                         normal_data=normal_data,
                         horizon=params['horizon'])
    return eval_data


def run(market_data, mongo_client, trading_sessions, factors_infos, params,
        category, instruments, begin_time, end_time):
    eval_results = run1(market_data=market_data,
                        trading_sessions=trading_sessions,
                        factors_infos=factors_infos,
                        params=params)
    pdb.set_trace()
    persist_evaluate_series(mongo_client=mongo_client,
                            eval_results=eval_results,
                            category=category,
                            code=INSTRUMENTS_CODES[instruments],
                            begin_time=begin_time,
                            end_time=end_time)


def run_source(fetch_market_func, mongo_client, trading_sessions,
               factors_infos, params, category, instruments, begin_time,
               end_time, start_time, adjusted_method):
    pdb.set_trace()
    market_data = fetch_market_func(instruments=instruments,
                                    begin_time=start_time,
                                    end_time=end_time,
                                    adjusted_method=adjusted_method,
                                    forced_alignment=True)

    run(market_data=market_data,
        mongo_client=mongo_client,
        trading_sessions=trading_sessions,
        factors_infos=factors_infos,
        params=params,
        category=category,
        instruments=instruments,
        begin_time=begin_time,
        end_time=end_time)


def start1(task_id, instruments, adjusted_method='pcr'):
    mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
    trading_sessions = (("21:00", "23:00"), ("09:00", "10:15"),
                        ("10:30", "11:30"), ("13:30", "15:00"))
    adjusted_method = 'pcr'
    begin_time = datetime.datetime(2026, 6, 1)
    end_time = datetime.datetime(2026, 6, 30)
    start_time = advanceDateByCalendar('china.sse', begin_time, '-1b')

    ###
    factors_infos, params = load_sirius_params(
        code=INSTRUMENTS_CODES[instruments], task_id=task_id)
    # factors_infos = [factors_infos[0]]

    source_configs = [
        #("bench", fetch_bench_data),
        ("research", fetch_research_data),
        #("trader", fetch_trader_data),
    ]

    for category, fetch_market_func in source_configs:
        run_source(fetch_market_func=fetch_market_func,
                   mongo_client=mongo_client,
                   trading_sessions=trading_sessions,
                   factors_infos=factors_infos,
                   params=params,
                   category=category,
                   instruments=instruments,
                   begin_time=begin_time,
                   end_time=end_time,
                   start_time=start_time,
                   adjusted_method=adjusted_method)

    # bench_market = fetch_bench_data(instruments=instruments,
    #                                       begin_time=start_time,
    #                                       end_time=end_time,
    #                                       adjusted_method=adjusted_method)

    # run(market_data=bench_market,
    #     mongo_client=mongo_client,
    #     trading_sessions=trading_sessions,
    #     factors_infos=factors_infos,
    #     params=params,
    #     category='bench',
    #     instruments=instruments,
    #     begin_time=begin_time,
    #     end_time=end_time)

    # research_market = fetch_research_data(instruments=instruments,
    #                                       begin_time=start_time,
    #                                       end_time=end_time,
    #                                       adjusted_method=adjusted_method)

    # run(market_data=research_market,
    #     mongo_client=mongo_client,
    #     trading_sessions=trading_sessions,
    #     factors_infos=factors_infos,
    #     params=params,
    #     category='research',
    #     instruments=instruments,
    #     begin_time=begin_time,
    #     end_time=end_time)

    # trader_market = fetch_trader_data(instruments=instruments,
    #                                   begin_time=start_time,
    #                                   end_time=end_time,
    #                                   adjusted_method=adjusted_method)

    # run(market_data=trader_market,
    #     mongo_client=mongo_client,
    #     trading_sessions=trading_sessions,
    #     factors_infos=factors_infos,
    #     params=params,
    #     category='trader',
    #     instruments=instruments,
    #     begin_time=begin_time,
    #     end_time=end_time)


if __name__ == '__main__':
    #pdb.set_trace()
    #td = pd.read_feather('/workspace/data/dev/kd/intelkit/records/raw_data/cn_futures/20260601/rb2610_20260601.feather')
    start1(instruments='rbb', task_id='1018806311332385')
