import itertools
from collections import namedtuple
from ultron.tradingday import *
from ultron.sentry.api import *
from lib.attr001.ftd001 import *
from lib.attr001.check001 import generate_bar_status
from lib.cux003 import FactorEvaluate1
from lumina.formual.impulse import Impulse
from lumina.evolution.fusion.actuator import Actuator

### 获取指定数据集
DEFAULT_PRICE = ['open', 'high', 'low', 'close', 'vwap']
DEFAULT_REL = ["volume", "value", "openint"]

## 成交量计算错误
DEFAILT_COVER = ["volume", "value", "vwap"]

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


def fetch_market_data(instruments,
                      begin_time,
                      end_time,
                      tick_size,
                      adjusted_method=None,
                      price_fields=DEFAULT_PRICE,
                      rel_fiedls=DEFAULT_REL,
                      cover_cols=DEFAILT_COVER):
    research_market = fetch_research_data(instruments=instruments,
                                          begin_time=begin_time,
                                          end_time=end_time,
                                          adjusted_method=adjusted_method)

    trader_market = fetch_trader_data(instruments=instruments,
                                      begin_time=begin_time,
                                      end_time=end_time,
                                      adjusted_method=adjusted_method)

    research_market, trader_market = algin_data2(research_market,
                                                 trader_market)

    for col in cover_cols:
        trader_market[col] = research_market[col]

    price_metrics = price_diff_metrics(research_market=research_market,
                                       trader_market=trader_market,
                                       tick_size=tick_size,
                                       price_fields=price_fields)

    rel_metrics = relative_diff_metrics(research_market=research_market,
                                        trader_market=trader_market,
                                        rel_fields=rel_fiedls)
    price_metrics = pd.DataFrame(price_metrics)
    rel_metrics = pd.DataFrame(rel_metrics)
    results = generate_bar_status(price_metrics, rel_metrics)

    return research_market, trader_market, {
        "price_metrics": price_metrics,
        "rel_metrics": rel_metrics,
        "results": results
    }


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