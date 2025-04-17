import sys, os, pdb
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath('../../..'))
#load_dotenv(os.path.join('../', '.env'))
load_dotenv()

from dichaos.agents.indexor.technical.agent import Agent
from dichaos.agents.indexor.porfolio import Portfolio
from dichaos.kdutils import kd_logger


def load_mirso(method):
    dirs = os.path.join(os.environ['BASE_PATH'], 'data', method, 'technical')

    filename = os.path.join(dirs, 'market_data.feather')
    market_data = pd.read_feather(filename)

    filename = os.path.join(dirs, 'returns_data.feather')
    returns_data = pd.read_feather(filename)
    total_data = market_data.merge(
        returns_data[['trade_date', 'code', 'ret_o2o']],
        on=['trade_date', 'code'],
        how='left')
    return total_data


def main(method, symbol, date):
    total_data = load_mirso(method)
    base_dir = 'brain/bsotr1/'
    source_path = os.path.join(os.environ['BASE_PATH'], base_dir, f'{symbol}_{date}')
    agent = Agent.load_checkpoint(source_path)
    pdb.set_trace()
    portfolio = Portfolio(symbol=symbol, lookback_window_size=2)

    total_data = total_data.set_index(['trade_date', 'code'])

    trend_data = agent.calculate_trend(total_data)
    format_trend_data = agent.format_trend(trend_data)

    trend1_dates = set(format_trend_data.keys())

    volatility_data = agent.calculate_volatility(total_data)
    format_volatility_data = agent.format_volatility(volatility_data)

    volatility1_dates = set(format_volatility_data[0].keys())
    volatility2_dates = set(format_volatility_data[1].keys())

    momentum_data = agent.calculate_momentum(total_data)
    format_momentum_data = agent.format_momentum(momentum_data)
    momentum_dates = set(format_momentum_data.keys())

    mean_reversion_data = agent.calculate_mean_reversion(total_data)
    format_mean_reversion_data = agent.format_mean_reversion(
        mean_reversion_data)

    mean_reversion1_dates = set(format_mean_reversion_data[0].keys())
    mean_reversion2_dates = set(format_mean_reversion_data[1].keys())
    mean_reversion3_dates = set(format_mean_reversion_data[2].keys())

    state_arb_data = agent.calculate_state_arb(total_data)
    format_state_arb_data = agent.format_state_arb(state_arb_data)

    state_arb1_dates = set(format_state_arb_data[0].keys())
    state_arb2_dates = set(format_state_arb_data[1].keys())

    format_future_data = agent.format_future(total_data['ret_o2o'])

    format_future_dates = set(format_future_data.keys())

    dates = trend1_dates & volatility1_dates & volatility2_dates & momentum_dates \
        & mean_reversion1_dates & mean_reversion2_dates & mean_reversion3_dates & \
            state_arb1_dates & state_arb2_dates & format_future_dates

    dates = list(dates)
    dates.sort()
    base_dir = './brain/aicso3/'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for d in dates:
        print('d:', d)
        agent.handing_data(trade_date=d,
                           symbol=symbol,
                           trend_data=format_trend_data,
                           volatility_data=format_volatility_data,
                           momentum_data=format_momentum_data,
                           mean_reversion_data=format_mean_reversion_data,
                           state_arb_data=format_state_arb_data)

        long_prompt, mid_prompt, short_prompt, reflection_prompt = agent.query_records(
            trade_date=d, symbol=symbol, top_k=5)

        portfolio.update_market_info(
            cur_date=d,
            market_price=total_data.loc[d, symbol]['price'],
            rets=total_data.loc[d, symbol]['ret_o2o'])

        response = agent.generate_prediction(
            trade_date=d,
            symbol=symbol,
            short_prompt=short_prompt,
            mid_prompt=mid_prompt,
            long_prompt=long_prompt,
            reflection_prompt=reflection_prompt)

        actions = agent.actions(response=response)
        kd_logger.info('response:{0}'.format(response))

        portfolio.record_action(action={'direction': actions})
        feedback = portfolio.feedback()

        try:
            if portfolio.is_refresh():
                feedback = portfolio.feedback()
                agent.update_memory(trade_date=d,
                                    symbol=symbol,
                                    response=response,
                                    feedback=feedback)
        except Exception as e:
            print('update memory error:', e)
            # 这里可能是因为没有反馈数据
            pass

        ## 保存记忆点
        #pdb.set_trace()
        
        path = os.path.join(os.environ['BASE_PATH'], f'brain/{method}/', f'{symbol}_{d}')
        agent.save_checkpoint(path=path)


main('bsotr1', 'IF', '2023-06-02')
