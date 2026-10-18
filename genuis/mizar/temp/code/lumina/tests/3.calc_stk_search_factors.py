import pdb
import numpy as np
from create_data import create_cnstock_box
from ultron.ump.metrics import grid_helper as GridHelper
from ultron.ump.metrics.grid_search import GridSearch
from lumina.factors.buy import MinFactorChaosBuyL, MinFactorChaosBuyS
from lumina.factors.sell import MinFactorAtrNStop, MinFactorChaosSell
from lumina.position.position_fake import FakePosition

factors = {
    'factor_dm': {
        'buy': {
            'class': [MinFactorChaosBuyL],
            'resample_max': [80],
            'resample_min': [5],
            'change_threshold': [0.08],
            'position': [{
                'class': FakePosition
            }]
        },
        'sell': {
            'class': [MinFactorChaosSell],
            'slow': [80],
            'fast': [5]
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

    #close_atr_range = np.arange(1.0, 1.5, 0.5)
    #pre_atr_range = np.arange(1.0, 2.0, 0.5)  #np.arange(1.0, 1.5, 0.5)

    #sell_atr_pre_factor_grid = {
    #    'class': [FactorPreAtrNStop],
    #    'pre_atr_n': pre_atr_range
    #}

    #print('暴跌保护止损参数pre_atr_n设置范围:{}'.format(pre_atr_range))
    #print('盈利保护止盈参数close_atr_n设置范围:{}'.format(close_atr_range))

    sell_factors_product = GridHelper.gen_factor_grid(
        GridHelper.K_GEN_FACTOR_PARAMS_SELL,
        [
            sell_atr_nstop_factor_grid,
            #sell_atr_pre_factor_grid
            factor_grid
        ],
        need_empty_sell=False)
    return sell_factors_product


def main():
    sell_factors_product = create_sell_factors(name='factor_dm')
    buy_factors_product = create_buy_factors(name='factor_dm')

    print('卖出因子参数共有{}种组合方式'.format(len(sell_factors_product)))
    #print('卖出因子组合0: 形式为{}'.format(sell_factors_product[0]))
    print('买入因子参数共有{}种组合方式'.format(len(buy_factors_product)))
    #print('买入因子组合形式为{}'.format(buy_factors_product))
    print('组合因子参数数量{}'.format(
        len(buy_factors_product) * len(sell_factors_product)))

    read_cash = 5000000
    benchmark_kl_pd, pick_kl_pd_dict, choice_symbols = create_cnstock_box()

    grid_search = GridSearch(read_cash,
                             choice_symbols,
                             benchmark_kl_pd=benchmark_kl_pd,
                             buy_factors_product=buy_factors_product,
                             sell_factors_product=sell_factors_product)

    grid_search.kl_pd_manager.set_pick_time(pick_kl_pd_dict)
    scores, score_tuple_array = grid_search.fit(n_jobs=4)
    print('最优参数组合为{}'.format(score_tuple_array))


main()
