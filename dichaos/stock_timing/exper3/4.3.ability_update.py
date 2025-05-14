import sys, os, pdb, datetime
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv
from alphacopilot.calendars.api import advanceDateByCalendar

sys.path.insert(0, os.path.abspath('../../..'))

load_dotenv()

from dichaos.agents.indexor.ability.agent import Agent
from dichaos.agents.indexor.ability.model import IndicatorList, KLine
from dichaos.kdutils import kd_logger
from kdutils.until import create_agent_path
from kdutils.fetch_data import fetch_main_daily
from kdutils.notice import feishu


def load_data(begin_date, end_date, code):
    ### 主力日频价格
    market_data = fetch_main_daily(begin_date, end_date, codes=[code])
    market_data['trade_date'] = pd.to_datetime(market_data['trade_date'])
    return market_data[(market_data.trade_date >= begin_date)
                       & (market_data.trade_date <= end_date)]


def main():
    method = 'aicso2'
    end_date = advanceDateByCalendar('china.sse', datetime.datetime.now(), '-1b')
    begin_date =  advanceDateByCalendar('china.sse', end_date, '-10b')
    code = 'IF'
    total_data = load_data(begin_date, end_date, code)
    path = create_agent_path(name=Agent.name,
                             method=method,
                             symbol=code,
                             date='2025-02-21')
    agent = Agent.load_checkpoint(path)
    total_data = total_data.set_index(['trade_date', 'code'])
    ## 技术指标
    sma5, sma10, sma20 = agent.calcuate_sma(total_data)

    ema12, ema26 = agent.calculate_ema(total_data)

    rsi_df = agent.calculate_rsi(total_data)

    macd_df = agent.calculate_macd(total_data)

    bollinger_df = agent.calculate_bollinger_bands(total_data)

    atr_df = agent.calculate_atr(total_data)

    vwap_df = agent.calculate_vwap(total_data)

    adx_df = agent.calculate_adx(total_data)

    obv_df = agent.calculate_obv(total_data)

    pp, r1, s1, r2, s2, r3, s3 = agent.calcuate_point(total_data)

    dates = sma5.index.get_level_values(0).intersection(
        sma10.index.get_level_values(0)
    ).intersection(sma20.index.get_level_values(0)).intersection(
        ema12.index.get_level_values(0)
    ).intersection(ema26.index.get_level_values(0)).intersection(
        rsi_df.index.get_level_values(0)).intersection(
            macd_df.index.get_level_values(0)).intersection(
                bollinger_df.index.get_level_values(0)).intersection(
                    atr_df.index.get_level_values(0)).intersection(
                        vwap_df.index.get_level_values(0)).intersection(
                            adx_df.index.get_level_values(0)).intersection(
                                obv_df.index.get_level_values(0)).intersection(
                                    pp.index.get_level_values(0)).intersection(
                                        r1.index.get_level_values(0)
                                    ).intersection(s1.index.get_level_values(
                                        0)).intersection(
                                            r2.index.get_level_values(0)
                                        ).intersection(
                                            s2.index.get_level_values(0)
                                        ).intersection(
                                            r3.index.get_level_values(0)
                                        ).intersection(
                                            s3.index.get_level_values(0))
    dates = [d.strftime('%Y-%m-%d') for d in dates]  #[0:100]
    dates.sort()
    date = end_date.strftime('%Y-%m-%d')
    kline = KLine(date=date,
                  symbol=code,
                  open=total_data.loc[(date, code), 'open'],
                  close=total_data.loc[(date, code), 'close'],
                  high=total_data.loc[(date, code), 'high'],
                  low=total_data.loc[(date, code), 'low'],
                  volume=total_data.loc[(date, code), 'volume'])

    indicator_list = IndicatorList(date=date)
    indicator_list.set_indicator(sma5=sma5.loc[date],
                                 sma10=sma10.loc[date],
                                 sma20=sma20.loc[date],
                                 ema12=ema12.loc[date],
                                 ema26=ema26.loc[date],
                                 rsi=rsi_df.loc[date],
                                 macd=macd_df.loc[date],
                                 atr=atr_df.loc[date],
                                 vwap=vwap_df.loc[date],
                                 adx=adx_df.loc[date],
                                 obv=obv_df.loc[date],
                                 pp=pp.loc[date],
                                 r1=r1.loc[date],
                                 s1=s1.loc[date],
                                 r2=r2.loc[date],
                                 s2=s2.loc[date],
                                 r3=r3.loc[date],
                                 s3=s3.loc[date])
    agent.handing_data(date, code, indicator_list, kline)
    short_prompt, reflection_prompt = agent.query_records(date, code)
    response = agent.generate_prediction(date=date,
                                         symbol=code,
                                         short_prompt=short_prompt,
                                         reflection_prompt=reflection_prompt)
    content = "{0}: 日期:{1}, 方向:{2}, 置信度:{3}, 推理原因:{4}".format(code,advanceDateByCalendar('china.sse', date, '1b').strftime('%Y-%m-%d'),response.signal,response.confidence,response.reasoning)
    print(content)
    feishu(content)
main()
