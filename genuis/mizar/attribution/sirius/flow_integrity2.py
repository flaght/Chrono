import datetime, itertools
from collections import namedtuple
from dotenv import load_dotenv

load_dotenv()

# from chaosmind.timing.sirius0001.workflow import WorkFlow
from chaosmind.timing.sirius0002.workflow import WorkFlow
from ultron.tradingday import *
from lib.uvx import load_sirius_params
from lib.attr001.ftd001 import *
from lib.attr001.integrity.inspect import build_horizon_returns
from lib.rl012.analysis import profitability, quantile, pred_metrics
from kdutils.data import fetch_metrics


class EvaluateTuple(
        namedtuple('EvaluateTuple',
                   ('name', 'raw_factors', 'raw_returns', 'factor_series',
                    'nxt1_ret_series', 'f_scaled_series', 'ic_series',
                    'gross_ret_series', 'turnover_series', 'net_ret_series',
                    'nav_series'))):
    __slots__ = ()


def run1(category, factors_infos, params, code, symbol, task_id, factors_data):
    pdb.set_trace()
    workflow = WorkFlow(directory=params['model_path'],
                        code=code,
                        symbol=symbol,
                        task_id=task_id,
                        factors_infos=factors_infos,
                        softmax_temperature=params['softmax_temperature'],
                        min_open_signal_abs=params['min_open_signal_abs'],
                        logit_clip=params['logit_clip'],
                        min_trade_advantage=params['min_trade_advantage'],
                        min_margin=params['min_margin'],
                        method=params['method'],
                        win=params['win'])
    total_data1 = factors_data.dropna()
    all_trade_times = total_data1.index.get_level_values(
        'trade_time').unique().sort_values()
    res = []
    for time in all_trade_times:
        print(time)
        rt = workflow.create_signals(trade_time=time, data=total_data1)
        res.append(rt)
    results = pd.DataFrame(res)
    pdb.set_trace()
    results['category'] = category
    return results


## 模型预测er
def start1(task_id, instruments):
    mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
    begin_time = datetime.datetime(2026, 5, 7)
    end_time = datetime.datetime(2026, 6, 5)
    start_time = advanceDateByCalendar('china.sse', begin_time, '-1b')
    end_time1 = advanceDateByCalendar('china.sse', end_time, '1b')

    factors_infos, params = load_sirius_params(
        code=INSTRUMENTS_CODES[instruments], task_id=task_id)

    category = ['bench', 'research', 'trader']
    ### 加载对应的因子
    names = [f['formula'] for f in factors_infos]
    factors_data = fetch_metrics(
        category=category,
        code=INSTRUMENTS_CODES[instruments],
        begin_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=end_time1.strftime("%Y-%m-%d %H:%M:%S"),
        names=names,
        table_name='raw_factors')
    pdb.set_trace()
    for cy in category:
        factors1 = factors_data[factors_data['category'] == cy]
        factors_data1 = factors1.pivot_table(index=['trade_time', 'code'],
                                             columns='name',
                                             values='value',
                                             aggfunc='last')
        results = run1(category=cy,
                       factors_infos=factors_infos,
                       params=params,
                       code=INSTRUMENTS_CODES[instruments],
                       symbol='rb9999',
                       task_id=task_id,
                       factors_data=factors_data1)

        update_netout_series1(mongo_client=mongo_client,
                              series_data=results,
                              table_name='realm_netout_series',
                              category=cy)


######### ===========>


def run_start(netout_data,
              returns_data,
              cost_rate,
              holding_period,
              pnl_method='points_norm',
              factor_name='value',
              return_name='future_ret_h'):
    df = netout_data.merge(returns_data,
                           on=['trade_time',
                               'code']).sort_values(by=['trade_time', 'code'])

    df1 = build_horizon_returns(df=df,
                                ret_col="nxt1_ret",
                                return_name=return_name,
                                holding_period=holding_period)

    _, profit_daily, _, _ = profitability(
        data=df1[['trade_time', factor_name, return_name]],
        factor_name=factor_name,
        return_name=return_name,
        cost_rate=cost_rate,
        max_pos=0,
        holding_period=holding_period,
        pnl_method=pnl_method,
    )

    spread_sequence, _ = quantile(
        data=df1[['trade_time', factor_name, return_name]],
        factor_name=factor_name,
        return_name=return_name,
    )
    spread_sequence.name = 'ic'
    metrics_daily = pd.concat([profit_daily, spread_sequence], axis=1)
    return metrics_daily


def run_source(mongo_client, params, category, instruments, cost_rate,
               begin_time, end_time, start_time, task_id):
    pdb.set_trace()
    netout_data = fetch_netout(
        category=category,
        code=INSTRUMENTS_CODES[instruments],
        begin_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
        table_name="netout_series",
        mongo_client=mongo_client)

    returns_data = fetch_metrics(
        category=category,
        code=INSTRUMENTS_CODES[instruments],
        begin_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
        names=None,
        table_name='raw_returns',
        mongo_client=mongo_client)

    returns_data = returns_data.drop(
        ['category'], axis=1).rename(columns={"value": 'nxt1_ret'})

    metrics_daily = run_start(returns_data=returns_data,
                              netout_data=netout_data,
                              holding_period=params['horizon'],
                              cost_rate=cost_rate)
    metrics_daily['task_id'] = task_id
    metrics_daily['category'] = category
    metrics_daily['code'] = INSTRUMENTS_CODES[instruments]

    update_netout_series2(
        mongo_client=mongo_client,
        df_data=metrics_daily.reset_index(),
        table_name='realm_netout_metrics',
        unique_keys=['trade_date', 'task_id', 'category', 'code'])


## 模型er 绩效评估
def start2(instruments, task_id):
    category = ['bench', 'research', 'trader']
    adjusted_method = 'pcr'

    mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
    begin_time = datetime.datetime(2026, 5, 7)
    end_time = datetime.datetime(2026, 6, 5)
    start_time = advanceDateByCalendar('china.sse', begin_time, '-1b')
    end_time1 = advanceDateByCalendar('china.sse', end_time, '1b')

    _, params = load_sirius_params(code=INSTRUMENTS_CODES[instruments],
                                   task_id=task_id)

    for category in category:
        run_source(mongo_client=mongo_client,
                   params=params,
                   category=category,
                   instruments=instruments,
                   cost_rate='1e-05',
                   begin_time=begin_time,
                   end_time=end_time1,
                   start_time=start_time,
                   task_id=task_id)


if __name__ == '__main__':
    start1(instruments='rbb', task_id='1029921127239410')
    start2(instruments='rbb', task_id='1029921127239410')
