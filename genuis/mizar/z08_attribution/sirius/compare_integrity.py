import datetime, pdb

from dotenv import load_dotenv

load_dotenv()

from lib.uvx import load_sirius_params
from lib.attr001.ftd001 import *
from lib.attr001.integrity import quality_diagnostics
from lib.attr001.integrity import market_diagnostics
from lib.attr001.integrity import impluse_diagnostics
from lib.attr001.integrity import fusion_diagnostics
from lib.attr001.integrity import evaluate_diagnostics
from lib.attr001.integrity import predict_diagnostics
from lib.attr001.integrity import analysis_diagnostics


def start1(task_id, instruments, tick_size):
    adjusted_method = None  #'pcr'
    price_fields = ['open', 'high', 'low', 'close', 'vwap']
    rel_fiedls = ["volume", "value", "openint"]
    cover_cols = ["volume", "value", "vwap"]
    trading_sessions = (("21:00", "23:00"), ("09:00", "10:15"),
                        ("10:30", "11:30"), ("13:30", "15:00"))

    begin_time = datetime.datetime(2026, 5, 20)
    end_time = datetime.datetime(2026, 5, 22)

    research_market = fetch_research_data(instruments=instruments,
                                          begin_time=begin_time,
                                          end_time=end_time,
                                          adjusted_method=adjusted_method)

    trader_market = fetch_trader_data(instruments=instruments,
                                      begin_time=begin_time,
                                      end_time=end_time,
                                      adjusted_method=adjusted_method)

    ### 指定收益率
    ### 做一个交集
    comm_index = research_market.index.intersection(trader_market.index)
    research_market = research_market.loc[comm_index]
    trader_market = trader_market.loc[comm_index]

    research_market = filter_trading_time(data=research_market,
                                          trading_sessions=trading_sessions)

    trader_market = filter_trading_time(data=trader_market,
                                        trading_sessions=trading_sessions)

    research_market = research_market.set_index(['trade_time', 'code'])
    trader_market = trader_market.set_index(['trade_time', 'code'])

    ### layer0 验证
    research_quality_metrics = quality_diagnostics(
        data=research_market,
        name='research',
        trading_sessions=trading_sessions)
    trader_quality_metrics = quality_diagnostics(
        data=trader_market, name='trader', trading_sessions=trading_sessions)

    ### layer1 验证
    market_metrics = market_diagnostics(research_data=research_market,
                                        trader_data=trader_market,
                                        adjusted_method=adjusted_method,
                                        tick_size=tick_size)

    print(research_quality_metrics)
    print(trader_quality_metrics)
    print(market_metrics)

    ### layer2 验证
    factors_infos, params = load_sirius_params(
        code=INSTRUMENTS_CODES[instruments], task_id=task_id)
    # research_matrix = market_data_format(market_data=research_market)
    # trader_matrix = market_data_format(market_data=trader_market)
    research_returns = create_returns(market_data=research_market,
                                      horizon=params['horizon'],
                                      name='close')
    trader_returns = create_returns(market_data=trader_market,
                                    horizon=params['horizon'],
                                    name='close')

    pdb.set_trace()

    ### layer2 因子值
    ### 基础字段计算
    impluse_metrics = impluse_diagnostics(factors_infos=factors_infos,
                                          research_data=research_market,
                                          trader_data=trader_market)

    ### 衍生字段计算
    original_metrics, normal_metrics = fusion_diagnostics(
        factors_infos=factors_infos,
        research_data=research_market,
        trader_data=trader_market)

    ### layer4 绩效比对
    ### 绩效评估
    research_normal_data = normal_metrics['research_factors'].reset_index(
    ).merge(research_returns, on=['trade_time', 'code'])
    trader_normal_data = normal_metrics['trader_factors'].reset_index().merge(
        trader_returns, on=['trade_time', 'code'])
    metrics_results = evaluate_diagnostics(research_data=research_normal_data,
                                           trader_data=trader_normal_data,
                                           factors_infos=factors_infos,
                                           params=params)

    ### laver5 预测值
    predict_metrics = predict_diagnostics(
        factors_infos=factors_infos,
        params=params,
        instruments=instruments,
        task_id=task_id,
        research_data=normal_metrics['research_factors'],
        trader_data=normal_metrics['trader_factors'])

    ### level6 预测值绩效
    analysis_metrics = analysis_diagnostics(
        research_data=predict_metrics['research_net_out'].merge(
            research_returns, on=['trade_time', 'code']),
        trader_data=predict_metrics['trader_net_out'].merge(
            trader_returns, on=['trade_time', 'code']),
        factor_name='value',
        return_name='future_ret_h',
        pnl_method='points_norm',
        cost_rate='1e-05',
        params=params)


if __name__ == '__main__':
    start1(instruments='rbb', tick_size=1, task_id='1029921127239410')
