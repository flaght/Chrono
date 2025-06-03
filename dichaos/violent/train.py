import sys, os, pdb, datetime
import pandas as pd
import numpy as np
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

from obility.agent import Agent
from dichaos.kdutils import kd_logger
from factors.builder import *
from factors.model import *


def create_data(start_date, symbol, n):
    trade_time = pd.date_range(start=start_date, periods=n, freq='B')
    # 模拟价格数据
    np.random.seed(42)
    price = np.cumsum(np.random.randn(n)) + 100  # 模拟价格走势

    open_price = price + np.random.randn(n) * 0.5
    close_price = price + np.random.randn(n) * 0.5
    high_price = np.maximum(open_price, close_price) + np.random.rand(n)
    low_price = np.minimum(open_price, close_price) - np.random.rand(n)

    # 模拟成交量和成交额
    volume = np.random.randint(1000, 10000, size=n)
    amount = volume * close_price

    # 计算收益率（对数收益率）
    returns = np.log(close_price / np.roll(close_price, 1))
    returns[0] = 0  # 第一行无法计算对数收益率

    # 构建 DataFrame
    total_data = pd.DataFrame({
        'trade_time': trade_time,
        'open': open_price.round(2),
        'high': high_price.round(2),
        'low': low_price.round(2),
        'close': close_price.round(2),
        'volume': volume,
        'amount': amount.round(2),
        'return': returns.round(4)
    })
    total_data['code'] = symbol
    total_data = total_data.set_index(['trade_time', 'code'])
    return total_data


def create_factors(start_date, total_data):
    begin_date = (datetime.datetime.strptime(start_date, '%Y-%m-%d') -
                  timedelta(days=30)).strftime('%Y-%m-%d')
    builder = Builder()
    sma5, sma10, sma20 = builder.calcuate_sma(data=total_data)
    ema12, ema26 = builder.calculate_ema(data=total_data)
    rsi = builder.calculate_rsi(data=total_data)
    macd = builder.calculate_macd(data=total_data)
    bollinger = builder.calculate_bollinger_bands(data=total_data)
    atr_pd = builder.calculate_atr(data=total_data)
    vwap = builder.calculate_vwap(data=total_data)
    adx_pd = builder.calculate_adx(data=total_data)
    obv_pd = builder.calculate_obv(data=total_data)
    pp, r1, s1, r2, s2, r3, s3 = builder.calcuate_point(data=total_data)
    return sma5.loc[start_date:], sma10.loc[start_date:], sma20.loc[start_date:], \
              ema12.loc[start_date:], ema26.loc[start_date:], rsi.loc[start_date:], \
                macd.loc[start_date:], bollinger.loc[start_date:], \
                atr_pd.loc[start_date:], vwap.loc[start_date:], adx_pd.loc[start_date:], \
                obv_pd.loc[start_date:], pp.loc[start_date:], r1.loc[start_date:], \
                s1.loc[start_date:], r2.loc[start_date:], s2.loc[start_date:], \
                r3.loc[start_date:], s3.loc[start_date:]


begin_date = '2024-01-01'
symbol = 'RB'

agent = Agent.from_config(path=os.path.join(Agent.name))
total_data = create_data(start_date=begin_date, symbol=symbol, n=100)
sma5, sma10, sma20, ema12, ema26, rsi, macd, bollinger, atr_pd, vwap, adx_pd, obv_pd, pp, r1, s1, r2, s2, r3, s3 = \
create_factors(start_date=begin_date, total_data=total_data)
dates = sma5.index.get_level_values(0)

for date1 in dates[3:]:
    kline = KLine(date=date1.strftime('%Y-%m-%d'),
                  symbol=symbol,
                  open=total_data.loc[date1, 'open'].values[0],
                  close=total_data.loc[date1, 'close'].values[0],
                  high=total_data.loc[date1, 'high'].values[0],
                  low=total_data.loc[date1, 'low'].values[0],
                  volume=total_data.loc[date1, 'volume'].values[0])
    indicator_list = IndicatorList(date=date1.strftime('%Y-%m-%d'))
    indicator_list.set_indicator(sma5=sma5.loc[date1],
                                 sma10=sma10.loc[date1],
                                 sma20=sma20.loc[date1],
                                 ema12=ema12.loc[date1],
                                 ema26=ema26.loc[date1],
                                 rsi=rsi.loc[date1],
                                 macd=macd.loc[date1],
                                 bollinger=bollinger.loc[date1],
                                 atr=atr_pd.loc[date1],
                                 vwap=vwap.loc[date1],
                                 adx=adx_pd.loc[date1],
                                 obv=obv_pd.loc[date1],
                                 pp=pp.loc[date1],
                                 r1=r1.loc[date1],
                                 s1=s1.loc[date1],
                                 r2=r2.loc[date1],
                                 s2=s2.loc[date1],
                                 r3=r3.loc[date1],
                                 s3=s3.loc[date1])

    future_data = total_data.loc[date1.strftime('%Y-%m-%d')]
    future_data = future_data.to_dict(orient='records')[0]

    agent.handing_data(trade_date=date1.strftime('%Y-%m-%d'),
                       symbol=symbol,
                       indicator_list=indicator_list,
                       kline=kline)
    short_prompt, reflection_prompt = agent.query_records(
        trade_date=date1.strftime('%Y-%m-%d'), symbol=symbol)
    response = agent.generate_suggestion(date=date,
                                         symbol=symbol,
                                         short_prompt=short_prompt,
                                         reflection_prompt=reflection_prompt,
                                         chg=round(future_data['return'], 4))
    kd_logger.info('response:{0}'.format(response))
    
    ### 更新记忆
    pdb.set_trace()
    agent.update_memory(trade_date=date1.strftime('%Y-%m-%d'),
                    symbol=symbol,
                    response=response,
                    feedback={'feedback': 1, 'date': date1.strftime('%Y-%m-%d')})
    agent.save_checkpoint(
        path=os.path.join(os.environ['BASE_PATH'], 'memory', agent.name,
                          date1.strftime('%Y-%m-%d')))
