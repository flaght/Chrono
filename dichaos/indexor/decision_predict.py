import sys, os, pdb
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

from decision.agent import Agent as DecisionAgent
from opsuwer.agent import Agent as OpsuwerAgent
from obility.agent import Agent as ObilityAgent
from risker.agent import Agent as RiskerAgent

from kdutils.until import create_agent_path
from dichaos.kdutils import kd_logger

from dichaos.agents.indexor.porfolio import Portfolio


def load_mirso(method):
    dirs = os.path.join('records', 'data', method, 'technical')
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
    pdb.set_trace()
    total_data = load_mirso(method)

    path = create_agent_path(name=DecisionAgent.name,
                             method=method,
                             symbol=symbol,
                             date='2010-04-21')

    agent = DecisionAgent.load_checkpoint(path)

    obility_path = create_agent_path(name=ObilityAgent.name,
                                     method=method,
                                     symbol=symbol,
                                     date='2010-07-06')
    obility_agent = ObilityAgent.load_checkpoint(obility_path)

    opsuwer_path = create_agent_path(name=OpsuwerAgent.name,
                                     method=method,
                                     symbol=symbol,
                                     date='2010-07-06')
    opsuwer_agent = OpsuwerAgent.load_checkpoint(opsuwer_path)

    risk_agent = RiskerAgent.from_config(path=os.path.join(RiskerAgent.name))

    agent.set_agents(obility_agent)
    agent.set_agents(opsuwer_agent)

    portfolio = Portfolio(symbol=symbol, lookback_window_size=1)

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

    porfolio_dict = {
        'cost_basis': 10000,
        'cash': 100000,
        'margin_requirement': 0.0,
        "position": {
            "long": 0,
            "short": 0
        }
    }

    for date in dates:
        kline_sets = agent.create_kline(date=date,
                                        symbol=symbol,
                                        open=total_data.loc[(date, symbol),
                                                            'open'],
                                        close=total_data.loc[(date, symbol),
                                                             'close'],
                                        high=total_data.loc[(date, symbol),
                                                            'high'],
                                        low=total_data.loc[(date, symbol),
                                                           'low'],
                                        volume=total_data.loc[(date, symbol),
                                                              'volume'])
        indicator_list_sets = agent.create_indicator_list(
            date=date,
            sma5=sma5.loc[date],
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
        future_data = total_data.loc[date]
        future_data = future_data.to_dict(orient='records')[0]
        agent.handing_data(date, symbol, indicator_list_sets, kline_sets)
        pdb.set_trace()
        short_prompt, reflection_prompt = agent.query_records(date, symbol)

        risk_agent.handing_data(
            date, symbol,
            risk_agent.create_kline(date=date,
                                    symbol=symbol,
                                    open=total_data.loc[(date, symbol),
                                                        'open'],
                                    close=total_data.loc[(date, symbol),
                                                         'close'],
                                    high=total_data.loc[(date, symbol),
                                                        'high'],
                                    low=total_data.loc[(date, symbol), 'low'],
                                    volume=total_data.loc[(date, symbol),
                                                          'volume']))
        short_data = risk_agent.query_records(date, symbol)

        risk_data = risk_agent.generate(date, symbol, short_data,
                                        porfolio_dict)
        portfolio.update_market_info(
            cur_date=date,
            market_price=total_data.loc[date, symbol]['close'],
            rets=total_data.loc[date, symbol]['ret_o2o'])

        response = agent.generate_prediction(
            date=date,
            symbol=symbol,
            short_prompt=short_prompt,
            reflection_prompt=reflection_prompt,
            risk_data=risk_data,
            portfolio=porfolio_dict)
        print(response)
        break


main('aicso2', 'IF')
