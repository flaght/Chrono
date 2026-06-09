import datetime, pdb, itertools
import pandas as pd
import numpy as np
# from lumina.formual.iactuator import Iactuator
from lumina.formual.impulse import Impulse
from lumina.evolution.fusion.actuator import Actuator
from alphacopilot.api.calendars import advanceDateByCalendar
from ultron.sentry.api import *
from dotenv import load_dotenv

load_dotenv()

from config.contract import INSTRUMENTS_CODES
from kdutils.data import fetch_main_market
from kdutils.tactix import Tactix
from lib.ret001 import create_chg, create_yields
from lib.uvx import load_sirius_params
from lib.cux003 import FactorEvaluate1


### 研究数据 采用主力合约
def fetch_research(instruments, days, task_id):
    pdb.set_trace()

    ## 加载策略属性
    factors_infos, params = load_sirius_params(
        code=INSTRUMENTS_CODES[instruments], task_id=task_id)
    pdb.set_trace()
    dependencies = [
        eval(formula['formula'])._dependency for formula in factors_infos
    ]
    dependencies = list(itertools.chain.from_iterable(dependencies))

    # end_date = advanceDateByCalendar('china.sse', datetime.datetime.now(),
    #                                  '-{0}b'.format(0)).strftime('%Y-%m-%d')

    # start_date = advanceDateByCalendar(
    #     'china.sse', end_date, '-{0}b'.format(days)).strftime('%Y-%m-%d')
    
    end_date = '2026-05-15'
    start_date = "2026-03-01"

    market_data = fetch_main_market(begin_date=start_date,
                                    end_date=end_date,
                                    codes=[INSTRUMENTS_CODES[instruments]],
                                    keep_symbol=True)
    
    market_data = market_data.set_index(['trade_time', 'code'])
    res = {}
    cols = [
        'close', 'high', 'low', 'open', 'value', 'volume', 'openint', 'chg',
        'price', 'vwap'
    ]

    for col in cols:
        res[col] = market_data[col].unstack()

    ### 引用库计算
    factors_data1 = Impulse(dependencies).batch(data=res)
    total_data = factors_data1.reset_index()
    actuator = Actuator(k_split=1)

    original_factors, normal_factors = actuator.calculate(
        factors_infos=factors_infos,
        total_data=total_data,
        method='roll_zscore',
        win=15)
    
    original_factors = original_factors.reset_index()
    original_factors['trade_time'] = pd.to_datetime(
        original_factors['trade_time'])

    normal_factors = normal_factors.reset_index()
    normal_factors['trade_time'] = pd.to_datetime(normal_factors['trade_time'])

    ## 收益率计算
    chg_data = create_chg(market_data.reset_index())
    returns_data = create_yields(data=chg_data.copy(),
                                 horizon=params['horizon'])

    returns_data = returns_data.loc[start_date:end_date]
    returns_data = returns_data.reset_index()
    returns_data['trade_time'] = pd.to_datetime(returns_data['trade_time'])
    returns_data = returns_data.sort_values(by=['trade_time', 'code'])

    ### 绩效评估 ### 使用原始值 直接评估和使用标准化后的值评估 误差是否大？
    # pdb.set_trace()
    # dt1 = original_factors.merge(returns_data, on=['trade_time', 'code']).dropna(subset=['nxt1_ret'])
    # evaluate1 = FactorEvaluate1(factor_data=dt1,
    #                             factor_name="MADecay(30,'oi039_1_2_1')",
    #                             ret_name="nxt1_ret",
    #                             roll_win=15,
    #                             fee=0.0,
    #                             scale_method="roll_zscore",
    #                             expression="MADecay(30,'oi039_1_2_1')",
    #                             resampling_win=5)
    # stats_dt1 = evaluate1.run()
    
    
    dt2 = normal_factors.merge(returns_data, on=['trade_time', 'code']).dropna(subset=['nxt1_ret'])
    
    for factor in factors_infos:
        pdb.set_trace()
        evaluate2 = FactorEvaluate1(factor_data=dt2,
                                factor_name=factor['formula'],
                                ret_name="nxt1_ret",
                                roll_win=15,
                                fee=0.0,
                                scale_method="raw",
                                expression=factor['formula'],
                                resampling_win=5)
    
        stats_dt2 = evaluate2.run()
        ### 存储阶段范围内因子值，IC曲线，收益曲线
        print(stats_dt2)



if __name__ == '__main__':
    # variant = Tactix().start()
    fetch_research(instruments='rbb', days=500, task_id='1029921127239410')
