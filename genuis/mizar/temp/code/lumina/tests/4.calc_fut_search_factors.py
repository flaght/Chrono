import pdb
import numpy as np
from create_data import create_cnfut_box
from ultron.env import *
from ultron.ump.core import env
from ultron.ump.metrics import grid_helper as GridHelper
from ultron.ump.metrics.grid_search import GridSearch
from lumina.factors.buy import MinFacotorBaMaBuyL, MinFacotorBaMaBuyS
from lumina.factors.buy import MinFactorBollBuyL, MinFactorBollBuyS
from lumina.factors.buy import MinFactorBollBandtBuyL, MinFactorBollBandtBuyS
from lumina.factors.buy import MinFactorSWWaveBuyL, MinFactorSWWaveBuyS
from lumina.factors.sell import MinFactorSWWaveSell
from lumina.factors.buy import MinFactorKingKeltnerBuyL, MinFactorKingKeltnerBuyS
from lumina.factors.sell import MinFactorKingKeltnerSell
from lumina.factors.buy import MinFactorICUBuyL, MinFactorICUBuyS
from lumina.factors.sell import MinFactorICUSell
from lumina.factors.buy import MinFactorGhostTraderBuyL, MinFactorGhostTraderBuyS
from lumina.factors.sell import MinFactorGhostTraderSell
from lumina.factors.buy import MinFactorDoubleMaBuyL, MinFactorDoubleMaBuyS
from lumina.factors.sell import MinFactorDoubleMaSell
from lumina.factors.buy import MinFactorChaosBuyL, MinFactorChaosBuyS
from lumina.factors.sell import MinFactorChaosSell
from lumina.factors.buy import MinFactorRSRSBuyL, MinFactorRSRSBuyS
from lumina.factors.sell import MinFactorRSRSSell

from lumina.factors.sell import MinFactorAtrNStop
from lumina.position.position_atr import AtrPosition

factors = {
    'factor_bama': {
        'buy': {
            'class': [MinFactorRSRSBuyL, MinFactorRSRSBuyS],
            'resample_max': [20],
            'resample_min': [5],
            'change_threshold': [0.18],
            'ewm': [1],
            'position': [{
                'class': AtrPosition
            }]
        },
        'sell': {
            'class': [MinFactorRSRSSell],
            'ewm': [1],
        }
    }
}


def create_buy_factors(name):
    factor_grid = factors[name]['buy']
    buy_factors_product = GridHelper.gen_factor_grid(
        GridHelper.K_GEN_FACTOR_PARAMS_BUY, [factor_grid])
    return buy_factors_product


def create_sell_factors(name):

    factor_grid = factors[name]['sell']

    # 平仓组合取值范围
    stop_win_range = np.arange(2.0, 2.5, 0.5)
    stop_loss_range = np.arange(0.5, 1., 0.5)

    sell_atr_nstop_factor_grid = {
        'class': [MinFactorAtrNStop],
        'stop_loss_n': stop_loss_range,
        'stop_win_n': stop_win_range
    }

    sell_factors_product = GridHelper.gen_factor_grid(
        GridHelper.K_GEN_FACTOR_PARAMS_SELL, [factor_grid],
        need_empty_sell=False)
    return sell_factors_product


def main():
    sell_factors_product = create_sell_factors(name='factor_bama')
    buy_factors_product = create_buy_factors(name='factor_bama')
    print('卖出因子参数共有{}种组合方式'.format(len(sell_factors_product)))
    #print('卖出因子组合0: 形式为{}'.format(sell_factors_product[0]))
    print('买入因子参数共有{}种组合方式'.format(len(buy_factors_product)))
    #print('买入因子组合形式为{}'.format(buy_factors_product))
    print('组合因子参数数量{}'.format(
        len(buy_factors_product) * len(sell_factors_product)))

    env.g_market_target = env.EMarketTargetType.E_MARKET_TARGET_FUTURES_CN
    env.g_enable_ml_feature = True
    market_trade_year = 252
    n_fold = 2
    read_cash = 5000000
    benchmark_kl_pd, pick_kl_pd_dict, choice_symbols = create_cnfut_box()
    grid_search = GridSearch(read_cash,
                             choice_symbols,
                             benchmark_kl_pd=benchmark_kl_pd,
                             buy_factors_product=buy_factors_product,
                             sell_factors_product=sell_factors_product)

    grid_search.kl_pd_manager.set_pick_time(pick_kl_pd_dict)
    scores, score_tuple_array = grid_search.fit(n_jobs=1)
    pdb.set_trace()
    print('最优参数组合为{}'.format(score_tuple_array))


main()
