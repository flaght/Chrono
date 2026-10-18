import pdb
from create_data import create_cnstock_box
from ultron.env import *
from ultron.ump.core.ump import run_loop_back
from ultron.ump.core import env
from ultron.ump.trade.ml_feature import append_user_feature
from ultron.ump.agent import manager

from lumina import env as lumina_env
from lumina.factors.buy import MinFactorChaosBuyL, MinFactorChaosBuyS
from lumina.factors.sell import MinFactorChaosSell
from lumina.position import KellyPosition
from lumina.agent import main_lbm

lumina_env.g_max_window = 3
env.g_market_target = env.EMarketTargetType.E_MARKET_TARGET_CN
env.g_enable_ml_feature = True

### 配置拦截模型

for featue in main_lbm.g_feature_list:
    append_user_feature(featue)

manager.clear_user_agent()
manager.g_enable_user_agent = True ## 拦截模型开启
manager.append_user_agent(main_lbm.MainLBM,
                          market_name=env.EMarketTargetType.E_MARKET_TARGET_CN)


def main():
    buy_factors = [{
        'class': MinFactorChaosBuyS,
        'xd': 42,
        'fast': -1,
        'slow': -1,
        'position': {
            'class': KellyPosition
        },
    }, {
        'class': MinFactorChaosBuyL,
        'xd': 42,
        'fast': -1,
        'slow': -1,
        'position': {
            'class': KellyPosition
        },
    }]

    sell_factors = [{'class': MinFactorChaosSell, 'fast': 3, 'slow': 5}]
    read_cash = 5000000
    benchmark_kl_pd, pick_kl_pd_dict, choice_symbols = create_cnstock_box()

    pdb.set_trace()
    result_tuple, kl_pd_manager = run_loop_back(
        read_cash=read_cash,
        buy_factors=buy_factors,
        sell_factors=sell_factors,
        benchmark_kl_pd=benchmark_kl_pd,
        pick_kl_pd_dict=pick_kl_pd_dict,
        choice_symbols=choice_symbols)
    pdb.set_trace()
    print(result_tuple)


main()
