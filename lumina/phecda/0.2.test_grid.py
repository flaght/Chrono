import numpy as np
import pdb
from lumina.factors.buy import MinFactorKingKeltnerBuyL, MinFactorKingKeltnerBuyS
from lumina.factors.sell import MinFactorKingKeltnerSell
from lumina.factors.buy import MinFactorBollBuyL, MinFactorBollBuyS
from lumina.factors.sell import MinFactorBollSell
from lumina.factors.buy import MinFactorChaosBuyL, MinFactorChaosBuyS
from lumina.factors.sell import MinFactorChaosSell
from lumina.factors.sell import MinFactorAtrNStop
from lumina.factors.sell import MinFactorCloseAtrNStop

from ultron.ump.metrics import grid_helper as GridHelper
from ultron.ump.metrics.grid_search import GridSearch

stop_win_range = np.arange(2.0, 4.5, 0.5)
stop_loss_range = np.arange(0.5, 2, 0.5)

sell_atr_nstop_factor_grid = {
    'class': [MinFactorAtrNStop],
    'stop_loss_n': stop_loss_range,
    'stop_win_n': stop_win_range
}

close_atr_range = np.arange(1.0, 4.0, 0.5)

sell_atr_close_factor_grid = {
    'class': [MinFactorCloseAtrNStop],
    'close_atr_n': close_atr_range
}

print('AbuFactorAtrNStop止盈参数stop_win_n设置范围:{}'.format(stop_win_range))
print('AbuFactorAtrNStop止损参数stop_loss_n设置范围:{}'.format(stop_loss_range))
print('盈利保护止盈参数close_atr_n设置范围:{}'.format(close_atr_range))

sell_factors_product = GridHelper.gen_factor_grid(
    GridHelper.K_GEN_FACTOR_PARAMS_SELL,
    [sell_atr_nstop_factor_grid, sell_atr_close_factor_grid],
    need_empty_sell=True)
print('卖出因子参数共有{}种组合方式'.format(len(sell_factors_product)))
print('卖出因子组合0: 形式为{}'.format(sell_factors_product[0]))

buy_s_boll_factor_grid = {
    'class': [MinFactorBollBuyS],
    'ma_xd': [10],
    'ewm': [0, 1],
    'change_threshold': [0.12],
    'roc': [3],
    'offset': [1.5]
}

buy_l_chaos_factor_grid = {
    'class': [MinFactorChaosBuyL],
    'fast': [3, 5],
    'slow': [10, 15],
    'ewm': [0, 1]
}

buy_factors_product = GridHelper.gen_factor_grid(
    GridHelper.K_GEN_FACTOR_PARAMS_BUY,
    [buy_s_boll_factor_grid, buy_l_chaos_factor_grid])

pdb.set_trace()
print('买入因子参数共有{}种组合方式'.format(len(buy_factors_product)))
print('买入因子组合形式为{}'.format(buy_factors_product))

print('组合因子参数数量{}'.format(len(buy_factors_product) * len(sell_factors_product) ))

for i in range(len(buy_factors_product)):
    print('买入因子组合{}: {}\n'.format(i, buy_factors_product[i]))