import pdb, os, json
from dotenv import load_dotenv

load_dotenv()
from kdutils.creator import create_test_data
from obility.calculate import *
from obility.agent import Agent
from obility.model import *

## portfilio

from portfolio import Portflio


def load_mirso(method):
    start_date = '2015-01-01'
    codes = ['A', 'B', 'C', 'D', 'E']
    total_data = create_test_data(start_date=start_date, symbols=codes, n=10)
    return total_data


def create_indicator(total_data):
    pdb.set_trace()
    data_pd = total_data.unstack()
    sma5 = calcuate_sma(df=data_pd, period=5)
    sma5 = sma5.stack()
    sma5.name = 'sma5'

    sma10 = calcuate_sma(df=data_pd, period=10)
    sma10 = sma10.stack()
    sma10.name = 'sma10'

    sma20 = calcuate_sma(df=data_pd, period=20)
    sma20 = sma20.stack()
    sma20.name = 'sma20'

    ema12 = calculate_ema(df=data_pd, period=12)
    ema12 = ema12.stack()
    ema12.name = 'ema12'

    ema26 = calculate_ema(df=data_pd, period=26)
    ema26 = ema26.stack()
    ema26.name = 'ema26'

    rsi = calculate_rsi(df=data_pd, period=14)
    rsi = rsi.stack()
    rsi.name = 'rsi'

    #macd, _, _ = calculate_macd(df=data_pd, fast_period=12, slow_period=26)

    vwap = calculate_vwap(df=data_pd)
    vwap = vwap.stack()
    vwap.name = 'vwap'

    pp, r1, s1, r2, s2, r3, s3 = calcuate_point(df=data_pd)

    pp = pp.stack()
    pp.name = 'pp'

    r1 = r1.stack()
    r1.name = 'r1'

    s1 = s1.stack()
    s1.name = 's1'

    r2 = r2.stack()
    r2.name = 'r2'

    s2 = s2.stack()
    s2.name = 's2'

    r3 = r3.stack()
    r3.name = 'r3'

    s3 = s3.stack()
    s3.name = 's3'

    ## 存储按照 trade_date + ticker--> 指标数
    indicator_pd = pd.concat([
        sma5, sma10, sma20, ema12, ema26, rsi, vwap, pp, r1, s1, r2, s2, r3, s3
    ],
                             axis=1)

    ##转字典
    indicator_pd = indicator_pd.reset_index()
    indicator_pd['trade_time'] = pd.to_datetime(
        indicator_pd['trade_time']).dt.strftime('%Y-%m-%d')
    indicator_pd = indicator_pd.to_dict(orient='records')
    return indicator_pd


def create_factors(indicator_pd, kline_pd):
    facotrs_set = []
    for indicator in indicator_pd:
        klines = kline_pd.loc[indicator['trade_time'],
                              indicator['code']].to_dict()

        kline_sets = KLine(code=indicator['code'],
                           date=indicator['trade_time'],
                           open=klines['open'],
                           close=klines['close'],
                           high=klines['high'],
                           low=klines['low'],
                           volume=klines['volume'],
                           amount=klines['amount'])

        indicator_sets = IndicatorSets(
            sma5=Indicator(name='sma5', id='sma5', values=indicator['sma5']),
            sma10=Indicator(name='sma10',
                            id='sma10',
                            values=indicator['sma10']),
            sma20=Indicator(name='sma20',
                            id='sma20',
                            values=indicator['sma20']),
            ema12=Indicator(name='ema12',
                            id='ema12',
                            values=indicator['ema12']),
            ema26=Indicator(name='ema26',
                            id='ema26',
                            values=indicator['ema26']),
            rsi=Indicator(name='rsi', id='rsi', values=indicator['rsi']),
            vwap=Indicator(name='vwap', id='vwap', values=indicator['vwap']),
            pp=Indicator(name='pp', id='pp', values=indicator['pp']),
            r1=Indicator(name='r1', id='r1', values=indicator['r1']),
            s1=Indicator(name='s1', id='s1', values=indicator['s1']),
            r2=Indicator(name='r2', id='r2', values=indicator['r2']),
            s2=Indicator(name='s2', id='s2', values=indicator['s2']),
            r3=Indicator(name='r3', id='r3', values=indicator['r3']),
            s3=Indicator(name='s3', id='s3', values=indicator['s3']),
            code=indicator['code'],
            date=indicator['trade_time'])

        facotrs_set.append(
            IndicatorList(code=indicator['code'],
                          date=indicator['trade_time'],
                          indicator=indicator_sets,
                          kline=kline_sets))

    return facotrs_set


def create_overview(futures_pd):
    overviews = {}
    futures_pd = futures_pd.reset_index()
    futures_pd['trade_time'] = pd.to_datetime(
        futures_pd['trade_time']).dt.strftime("%Y-%m-%d")
    grouped = futures_pd.groupby('trade_time')
    for k, g in grouped:
        g1 = g.to_dict(orient='records')
        res = []
        for g2 in g1:
            mo = MarketOverview(**{
                'code': g2['code'],
                'limit': 1,
                'chg': g2['return']
            })
            res.append(mo)
        overviews[k] = MarketOverviewList(date=k, stocks=res)
    return overviews


def train(trade_date, total_data, overview_pd, portflio, agent, codes):
    ## 转化为动作
    actions = portflio.transform(
        returns=total_data.loc[trade_date][['return']].reset_index().to_dict(
            orient='records'))

    ## 更新市场信息
    short_memory, reflection_memory = agent.query_memory(trade_date=trade_date,
                                                         codes=codes)
    overview_memory = overview_pd[trade_date].to_json()

    market_data = total_data.loc[trade_date][[
        'open', 'close', 'high', 'low'
    ]].reset_index().to_dict(orient='records')
    market_data = dict(
        zip([md['code'] for md in market_data],
            [md['code'] for md in market_data]))
    returns_data = total_data.loc[trade_date][[
        'return'
    ]].reset_index().to_dict(orient='records')
    pdb.set_trace()
    returns_data = dict(
        zip([md['code'] for md in returns_data],
            [md['return'] for md in returns_data]))

    portflio.update_market_info(cur_date=trade_date,
                                market_price=market_data,
                                returns=returns_data)

    portflio.record_action(cur_date=trade_date, actions=actions)

    feedback_res = portflio.feedback(cur_date=trade_date)
    pdb.set_trace()
    response = agent.generate_suggestion(
        date=trade_date,
        short_prompt=json.dumps(short_memory, ensure_ascii=False),
        reflection_prompt=json.dumps(reflection_memory, ensure_ascii=False),
        overview_memory=json.dumps(overview_memory, ensure_ascii=False))

    agent.update_memory(trade_date=trade_date,
                        response=response,
                        feedback=feedback_res)

    agent.save_checkpoint(path=os.path.join(os.environ['BASE_PATH'], 'memory',
                                            agent.name, trade_date))


def main(method):
    pdb.set_trace()
    total_data = load_mirso(method)
    overview_pd = create_overview(futures_pd=total_data['return'])
    indicator_pd = create_indicator(total_data=total_data)

    portflio = Portflio()
    ##K线转字典
    codes = total_data.index.get_level_values(level='code').unique().tolist()
    kline_pd = total_data

    agent = Agent.from_config(path=os.path.join(Agent.name))

    facotrs_set = create_factors(indicator_pd, kline_pd)
    ## 塞入 记忆库
    for factor in facotrs_set:
        agent.handing_filling(trade_date=factor.date,
                              code=factor.code,
                              indicator=factor.indicator,
                              kline=factor.kline)

    pdb.set_trace()
    dates = ['2015-01-02', '2015-01-05', '2015-01-06', '2015-01-07']
    for d in dates:
        train(trade_date=d,
              total_data=total_data,
              overview_pd=overview_pd,
              portflio=portflio,
              agent=agent,
              codes=codes)

    #train(trade_date='2015-01-05',
    #      total_data=total_data,
    #      overview_pd=overview_pd,
    #      portflio=portflio,
    #      agent=agent,
    #      codes=codes)

    #train(trade_date='2015-01-13',
    #      total_data=total_data,
    #      overview_pd=overview_pd,
    #      portflio=portflio,
    #      agent=agent,
    #      codes=codes)


main('ast')
