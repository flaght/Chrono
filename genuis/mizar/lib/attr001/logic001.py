from lib.attr001.ftd001 import *
from lib.attr001.check001 import generate_bar_status

### 获取指定数据集
DEFAULT_PRICE = ['open', 'high', 'low', 'close', 'vwap']
DEFAULT_REL = ["volume", "value", "openint"]

## 成交量计算错误
DEFAILT_COVER = ["volume", "value", "vwap"]


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
