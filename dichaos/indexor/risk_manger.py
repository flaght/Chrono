### 风控专家

import sys, os, pdb
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

from risker.agent import Agent
from risker.model import KLine, Risker
from dichaos.agents.indexor.porfolio import Portfolio
from dichaos.kdutils import kd_logger
from kdutils.until import create_agent_path


def load_mirso(method):
    dirs = os.path.join('records', 'data', method, 'technical')
    pdb.set_trace()
    filename = os.path.join(dirs, 'market_data.feather')
    market_data = pd.read_feather(filename)

    filename = os.path.join(dirs, 'returns_data.feather')
    returns_data = pd.read_feather(filename)
    total_data = market_data.merge(
        returns_data[['trade_date', 'code', 'ret_o2o']],
        on=['trade_date', 'code'],
        how='left')
    total_data['ret_o2o'] = total_data['ret_o2o'].astype('float32').round(4)
    return total_data


def main(method, symbol):
    total_data = load_mirso(method)
    agent = Agent.from_config(path=os.path.join(Agent.name))
    total_data = total_data.set_index(['trade_date', 'code'])
    pdb.set_trace()
    dates = total_data.index.get_level_values(0).unique().tolist()
    dates.sort()
    for date in dates:
        kline = KLine(date=date.strftime('%Y-%m-%d'),
                      symbol=symbol,
                      open=total_data.loc[(date, symbol), 'open'],
                      close=total_data.loc[(date, symbol), 'close'],
                      high=total_data.loc[(date, symbol), 'high'],
                      low=total_data.loc[(date, symbol), 'low'],
                      volume=total_data.loc[(date, symbol), 'volume'])

        porfolio = {
            'cost_basis': 10000,
            'cash': 100000,
            'margin_requirement': 0.0
        }
        pdb.set_trace()
        agent.handing_data(date, symbol, kline)
        short_data = agent.query_records(date, symbol)
        agent.generate(date, symbol, short_data, porfolio)

main('aicso2', 'IF')
