### 选择因子绩效跟踪，入库到MongoDB
import os, pdb
import datetime, itertools
from pathlib import Path
from collections import namedtuple
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from lib.attr001.ftd001 import *
from lib.attr001.logic001 import *
from lib.attr001.ftd003 import *
from kdutils.macro2 import *


def load_factor(stage="scope"):
    dirs1 = os.path.join("attribution", "sirius", stage)
    file_path = Path(dirs1)
    res = []
    for feat_file in file_path.rglob('*.csv'):
        factor_info = pd.read_csv(feat_file)
        name = feat_file.parts[-1].split('.')[0]
        factor_info['instrument'] = name  #.upper()
        res.append(factor_info)

    factor_infos = pd.concat(res, axis=0)
    return factor_infos


def persist_evaluate_series(mongo_client,
                            eval_results,
                            code,
                            category=None,
                            begin_time=None,
                            end_time=None,
                            stage='scope'):

    for eval_result in eval_results:
        print(code, eval_result.name)
        # delete_dataset(mongo_client=mongo_client,
        #                table_name="mizar_{0}_raw_factors".format(stage),
        #                code=code,
        #                name=eval_result.name)

        # delete_dataset(mongo_client=mongo_client,
        #                table_name='mizar_{0}_factors_metrics'.format(stage),
        #                code=code,
        #                name=eval_result.name)

        raw_factors = clip_series_to_window(eval_result.raw_factors,
                                            begin_time=begin_time,
                                            end_time=end_time)
        
        insert_factor_series1(series_data=raw_factors,
                              factor_name=eval_result.name,
                              code=code)
        # insert_full_series(mongo_client=mongo_client,
        #                    series_data=raw_factors,
        #                    table_name="mizar_{0}_raw_factors".format(stage),
        #                    factor_name=eval_result.name,
        #                    code=code)
        # update_evaluate_series(
        #     mongo_client=mongo_client,
        #     series_data=raw_factors,
        #     table_name="mizar_{0}_raw_factors".format(stage),
        #     factor_name=eval_result.name,
        #     code=code)

        df = clip_series_to_window(eval_result.resample_data,
                                   begin_time=begin_time,
                                   end_time=end_time)
        df = df.reset_index()
        df = df.drop([eval_result.name], axis=1)
        df['name'] = eval_result.name
        df['code'] = code

        insert_metrics_data(df_data=df)
        # insert_full_dataframe(
        #     mongo_client=mongo_client,
        #     df_data=df,
        #     table_name='mizar_{0}_factors_metrics'.format(stage),
        #     batch_size=10000)

        # update_netout_series2(
        #     mongo_client,
        #     df_data=df,
        #     table_name='mizar_{0}_factors_metrics'.format(stage),
        #     unique_keys=['trade_time', 'name', 'code'])

    print("returns {0}".format(code))
    insert_returns_series1(series_data=eval_result._asdict()['raw_returns'],
                           code=code)
    # delete_dataset(mongo_client=mongo_client,
    #                table_name="mizar_{0}_raw_returns".format(stage),
    #                code=code)
    # insert_full_series(mongo_client=mongo_client,
    #                    series_data=eval_result._asdict()['raw_returns'],
    #                    table_name="mizar_{0}_raw_returns".format(stage),
    #                    factor_name=eval_result.name,
    #                    code=code)
    # update_returns_series(mongo_client=mongo_client,
    #                       series_data=eval_result._asdict()['raw_returns'],
    #                       table_name=f"mizar_{0}_raw_returns".format(stage),
    #                       code=code,
    #                       category=category)


def persist_elite_factor(mongo_client, code, factor_infos):
    update_factor_infos(mongo_client=mongo_client,
                        factor_infos=factor_infos,
                        code=code,
                        table_name=f"mizar_elite_factor_info")


def _run1(instruments,
          market_data,
          trading_sessions,
          factors_infos,
          params,
          begin_pos=32):
    market_data = filter_trading_time(data=market_data,
                                      trading_sessions=trading_sessions)
    market_data = market_data.set_index(['trade_time', 'code'])
    
    returns_data = create_returns(
        market_data=market_data,
        horizon=params['horizon'],
        name=RETURN_NAME_MAPPING[INSTRUMENTS_CODES[instruments]])

    market_unstack = market_data_format(market_data)
    ## 创建基础字段
    impulse_factors = create_impulse(factors_infos=factors_infos,
                                     market_unstack=market_unstack)

    ## 衍生计算， 标准化
    total_data = pd.concat(
        [impulse_factors, market_data],
        axis=1).reset_index().sort_values(by=['trade_time', 'code'])
    actuator = Actuator(k_split=1)
    ### 原值调整过方向，然后时序标准化
    original_factors, normal_factors = actuator.calculate(
        factors_infos=factors_infos,
        total_data=total_data,
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


def run1(instruments, begin_time, end_time, factors_infos):
    mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
    params = {'horizon': 5, 'method': 'roll_zscore', 'win': 15}
    factors_infos = factors_infos.to_dict(orient='records')
    market_data = fetch_quant_data(instruments=instruments,
                                   begin_time=begin_time,
                                   end_time=end_time,
                                   adjusted_method='pcr')
    
    trading_sessions = TRADING_TIME_MAPPING[INSTRUMENTS_CODES[instruments]]
    eval_results = _run1(instruments=instruments,
                         market_data=market_data,
                         trading_sessions=trading_sessions,
                         factors_infos=factors_infos,
                         params=params)
    persist_evaluate_series(mongo_client=mongo_client,
                            eval_results=eval_results,
                            code=INSTRUMENTS_CODES[instruments],
                            begin_time=begin_time,
                            end_time=end_time)


def run2(instruments, factor_infos):
    mongo_client = MongoDBManager(uri=os.environ['MG_URI'])
    insert_elite_factor(factor_infos=factor_infos,
                        code=INSTRUMENTS_CODES[instruments])
    # persist_elite_factor(mongo_client=mongo_client,
    #                      code=INSTRUMENTS_CODES[instruments],
    #                      factor_infos=factor_infos)


##
def start1():
    begin_time = '2021-11-25'  #'2021-11-25'  #'2024-08-23'
    end_time = '2026-08-20'
    factor_infos = load_factor(stage='scope')

    instruments = factor_infos['instrument'].unique().tolist()
    for instrument in instruments:
        run1(instruments=instrument,
             begin_time=begin_time,
             end_time=end_time,
             factors_infos=factor_infos[factor_infos['instrument'] ==
                                        instrument])


def start2():
    factor_infos = load_factor(stage='elite')
    instruments = factor_infos['instrument'].unique().tolist()
    for instrument in instruments:
        run2(instruments=instrument,
             factor_infos=factor_infos[factor_infos['instrument'] ==
                                       instrument])


if __name__ == '__main__':
    start1()
