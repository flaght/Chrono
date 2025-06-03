import sys, os, pdb
import pandas as pd
import numpy as np
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

from obility.agent import Agent
from obility.model import IndicatorList, KLine
from dichaos.agents.indexor.porfolio import Portfolio
from dichaos.kdutils import kd_logger

symbol = 'RB'


### 构建模拟数据
n = 30
start_date = "2024-06-01"
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
total_data.tail()


### 加载已经训练的agent
agent = Agent.load_checkpoint(path=os.path.join(os.environ['BASE_PATH'], 'memory', Agent.name, '2024-06-07'))

### 计算策略依赖的技术指标
rsi_df = agent.calculate_rsi(total_data)

macd_df = agent.calculate_macd(total_data)


macd_df.tail()

## 用于回测
portfolio = Portfolio(symbol=symbol, lookback_window_size=0)



trade_time = '2024-06-14'


### K线数据
kline = KLine(date=trade_time,
                      symbol=symbol,
                      open=total_data.loc[(trade_time, symbol), 'open'],
                      close=total_data.loc[(trade_time, symbol), 'close'],
                      high=total_data.loc[(trade_time, symbol), 'high'],
                      low=total_data.loc[(trade_time, symbol), 'low'],
                      volume=total_data.loc[(trade_time, symbol), 'volume'])

print(kline.format())


### 技术指标管理类
indicator_list = IndicatorList(date=trade_time)
indicator_list.set_indicator(rsi=rsi_df.loc[trade_time],
                             macd=macd_df.loc[trade_time])

print(indicator_list.format())


### 存入记忆池
pdb.set_trace()
agent.handing_data(trade_time, symbol, indicator_list, kline)


### 获取记忆池中的数据
pdb.set_trace()
short_prompt, reflection_prompt = agent.query_records(trade_time, symbol)

print
