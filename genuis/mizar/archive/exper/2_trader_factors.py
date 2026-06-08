import itertools, datetime, pdb
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from ultron.sentry.api import *
from lumina.formual.impulse import Impulse
from lumina.evolution.fusion.actuator import Actuator
from config.contract import INSTRUMENTS_CODES
from lib.uvx import load_sirius_params
from lib.ret001 import create_chg, create_yields
from lib.cux003 import FactorEvaluate1
from lib.attr001.logic001 import *


def create_returns(market_data, horizon, name='vwap'):
    chg_data = create_chg(market_data.reset_index(), name)
    returns_data = create_yields(data=chg_data.copy(), horizon=horizon)
    returns_data = returns_data.reset_index()
    returns_data['trade_time'] = pd.to_datetime(returns_data['trade_time'])
    returns_data = returns_data.sort_values(by=['trade_time', 'code'])
    return returns_data


def _run_factors1(begin_time,
                  end_time,
                  instruments,
                  task_id,
                  factors_infos,
                  market_res=None,
                  fetch_market_func=None):

    if factors_infos is None:
        factors_infos, _ = load_sirius_params(
            code=INSTRUMENTS_CODES[instruments], task_id=task_id)
        factors_infos = [{
            'formula': "MADecay(30,'oi039_1_2_1')",
            'direction': 1
        }]

    dependencies = [
        eval(formula['formula'])._dependency for formula in factors_infos
    ]
    dependencies = list(itertools.chain.from_iterable(dependencies))

    if market_res is None:
        _, market_res = fetch_market_func(instruments=instruments,
                                          begin_time=begin_time,
                                          end_time=end_time)

    factors_data1 = Impulse(dependencies).batch(data=market_res)
    return factors_data1


def _run_factors2(begin_time,
                  end_time,
                  instruments,
                  task_id,
                  factors_infos,
                  market_res=None,
                  fetch_market_func=None):

    if factors_infos is None:
        factors_infos, _ = load_sirius_params(
            code=INSTRUMENTS_CODES[instruments], task_id=task_id)
        factors_infos = [{
            'formula': "MADecay(30,'oi039_1_2_1')",
            'direction': 1
        }]

    dependencies = [
        eval(formula['formula'])._dependency for formula in factors_infos
    ]
    dependencies = list(itertools.chain.from_iterable(dependencies))

    if market_res is None:
        _, market_res = fetch_market_func(instruments=instruments,
                                          begin_time=begin_time,
                                          end_time=end_time)

    factors_data1 = Impulse(dependencies).batch(data=market_res)
    total_data = factors_data1.reset_index()
    pdb.set_trace()
    actuator = Actuator(k_split=1)

    original_factors, normal_factors = actuator.calculate(
        factors_infos=factors_infos,
        total_data=total_data,
        method='roll_zscore',
        win=15)

    new_trade_time = pd.to_datetime(
        original_factors.index.get_level_values('trade_time'))
    original_factors.index = pd.MultiIndex.from_arrays(
        [new_trade_time,
         original_factors.index.get_level_values('code')])

    new_trade_time = pd.to_datetime(
        normal_factors.index.get_level_values('trade_time'))
    normal_factors.index = pd.MultiIndex.from_arrays(
        [new_trade_time,
         normal_factors.index.get_level_values('code')])

    return original_factors, normal_factors


def factor_metrics(factor_research,
                   factor_trade,
                   name,
                   eps: float = 1e-12,
                   upper: float = None,
                   lower: float = None,
                   threshold: float = None):
    diff = factor_research - factor_trade
    abs_diff = diff.abs()
    denom = pd.concat([
        factor_research.abs(),
        factor_trade.abs(),
        pd.Series(eps, index=factor_research.index)
    ],
                      axis=1).max(axis=1)
    rel_diff = abs_diff / denom
    pearson_corr = factor_research.corr(factor_trade, method="pearson")
    spearman_corr = factor_research.corr(factor_trade, method="spearman")
    sign_match = np.sign(factor_research) == np.sign(factor_trade)
    zero_cross = (factor_research * factor_trade) < 0

    res = {
        "valid_count": len(factor_research),
        "mean_abs_diff": abs_diff.mean(),
        "median_abs_diff": abs_diff.median(),
        "p95_abs_diff": abs_diff.quantile(0.95),
        "p99_abs_diff": abs_diff.quantile(0.99),
        "max_abs_diff": abs_diff.max(),
        "mean_rel_diff": rel_diff.mean(),
        "median_rel_diff": rel_diff.median(),
        "p95_rel_diff": rel_diff.quantile(0.95),
        "p99_rel_diff": rel_diff.quantile(0.99),
        "max_rel_diff": rel_diff.max(),
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "sign_match_ratio": sign_match.mean(),
        "zero_cross_ratio": zero_cross.mean(),
    }

    if threshold is not None:
        signal_r = factor_research > threshold
        signal_t = factor_trade > threshold

        res["signal_match_ratio"] = (signal_r == signal_t).mean()
        res["signal_flip_ratio"] = (signal_r != signal_t).mean()

    if upper is not None and lower is not None:
        signal_r = pd.Series(0, index=factor_research.index)
        signal_t = pd.Series(0, index=factor_trade.index)

        signal_r[factor_research > upper] = 1
        signal_r[factor_research < lower] = -1

        signal_t[factor_trade > upper] = 1
        signal_t[factor_trade < lower] = -1

        res["signal_match_ratio"] = (signal_r == signal_t).mean()
        res["signal_flip_ratio"] = (signal_r != signal_t).mean()
        res["long_short_reverse_ratio"] = ((signal_r * signal_t) == -1).mean()

    res['name'] = name
    return res


## 因子值对比
def start1(instruments, task_id, tick_size=1):
    
    price_fields = ['open', 'high', 'low', 'close', 'vwap']
    rel_fiedls = ["volume", "value", "openint"]
    cover_cols=["volume", "value", "vwap"]
    
    
    
    adjusted_method = None
    begin_time = datetime.datetime(2026, 5, 6)
    end_time = datetime.datetime(2026, 5, 13)

    research_market, trader_market, metrics_data = fetch_market_data(
        instruments=instruments,
        begin_time=begin_time,
        end_time=end_time,
        tick_size=tick_size,
        adjusted_method=adjusted_method,
        price_fields=price_fields,
        rel_fiedls=rel_fiedls,
        cover_cols=cover_cols)

    print(metrics_data['results']["field_status"])
    
    research_res = market_data_format(research_market)
    trader_res = market_data_format(trader_market)


    research_factors = _run_factors1(begin_time=begin_time,
                                     end_time=end_time,
                                     instruments=instruments,
                                     task_id=task_id,
                                     market_res=research_res,
                                     fetch_market_func=None,
                                     factors_infos=None)

    trader_factors = _run_factors1(begin_time=begin_time,
                                   end_time=end_time,
                                   instruments=instruments,
                                   task_id=task_id,
                                   market_res=trader_res,
                                   fetch_market_func=None,
                                   factors_infos=None)

    comm_index = research_factors.index.intersection(
        trader_factors.index)

    research_factors = research_factors.loc[comm_index]
    trader_factors = trader_factors.loc[comm_index]
    
    res = []
    for name in research_factors.columns:
        rt = factor_metrics(name=name,
                            factor_research=research_factors[name],
                            factor_trade=trader_factors[name])
        res.append(rt)
    results = pd.DataFrame(res)
    pdb.set_trace()
    print('--->')


def start2(instruments, task_id):
    method = None
    # begin_time = datetime.datetime(2026, 5, 20)
    # end_time = datetime.datetime(2026, 5, 20)
    begin_time = datetime.datetime(2026, 5, 13)
    end_time = datetime.datetime(2026, 5, 20)

    factors_infos, params = load_sirius_params(
        code=INSTRUMENTS_CODES[instruments], task_id=task_id)
    #factors_infos = [{'formula': "MDIFF(60,'iv012_2_3_1')", 'direction': 1}]

    research_data = fetch_research_data(instruments=instruments,
                                        begin_time=begin_time,
                                        end_time=end_time,
                                        adjusted_method=method)
    trader_data = fetch_trader_data(instruments=instruments,
                                    begin_time=begin_time,
                                    end_time=end_time,
                                    adjusted_method=method)

    ## 数据对齐
    research_data, trader_data = algin_data2(research_data, trader_data)

    ## 之前数据弄错了， 暂时设置为一样。目前已经修改和文华财经一致

    # cols = [
    #     'close', 'high', 'low', 'open', 'value', 'volume', 'openint', 'chg',
    #     'vwap'
    # ]

    cols = ['value', 'volume', 'openint']
    for col in cols:
        trader_data[col] = research_data[col]

    research_res = market_data_format(research_data)
    trader_res = market_data_format(trader_data)

    ## 收益率计算
    research_returns = create_returns(market_data=research_data,
                                      horizon=params['horizon'],
                                      name='close')
    trader_returns = create_returns(market_data=trader_data,
                                    horizon=params['horizon'],
                                    name='close')

    original_research_factors, normal_research_factors = _run_factors(
        begin_time=begin_time,
        end_time=end_time,
        instruments=instruments,
        task_id=task_id,
        market_res=research_res,
        factors_infos=factors_infos,
        fetch_market_func=fetch_research_data)

    original_trader_factors, normal_trader_factors = _run_factors(
        begin_time=begin_time,
        end_time=end_time,
        instruments=instruments,
        task_id=task_id,
        market_res=trader_res,
        factors_infos=factors_infos,
        fetch_market_func=fetch_trader_data)

    original_index = original_research_factors.index.intersection(
        original_trader_factors.index)
    research_index = normal_research_factors.index.intersection(
        normal_trader_factors.index)

    original_research_factors = original_research_factors.loc[original_index]
    original_trader_factors = original_trader_factors.loc[original_index]

    normal_research_factors = normal_research_factors.loc[research_index]
    normal_trader_factors = normal_trader_factors.loc[research_index]

    normal_research_data = normal_research_factors.reset_index().merge(
        research_returns, on=['trade_time', 'code']).dropna()
    normal_trader_data = normal_trader_factors.reset_index().merge(
        trader_returns, on=['trade_time', 'code']).dropna()

    normal_research_data = normal_research_data.set_index(
        ['trade_time', 'code'])
    normal_trader_data = normal_trader_data.set_index(['trade_time', 'code'])

    comm_index = normal_research_data.index.intersection(
        normal_trader_data.index)
    normal_trader_data = normal_trader_data.loc[comm_index].reset_index()
    normal_research_data = normal_research_data.loc[comm_index].reset_index()
    ###
    for factor in factors_infos:
        pdb.set_trace()
        research_evaluate = FactorEvaluate1(factor_data=normal_research_data,
                                            factor_name=factor['formula'],
                                            ret_name="nxt1_ret",
                                            roll_win=15,
                                            fee=0.0,
                                            scale_method="raw",
                                            expression=factor['formula'],
                                            resampling_win=params['horizon'])

        research_dt2 = research_evaluate.run()

        trader_evaluate = FactorEvaluate1(factor_data=normal_trader_data,
                                          factor_name=factor['formula'],
                                          ret_name="nxt1_ret",
                                          roll_win=15,
                                          fee=0.0,
                                          scale_method="raw",
                                          expression=factor['formula'],
                                          resampling_win=params['horizon'])

        trader_dt2 = trader_evaluate.run()

        print(research_dt2)
        print(trader_dt2)
        print('-->')


if __name__ == '__main__':
    start1(instruments='rbb', task_id='1029921127239410')
