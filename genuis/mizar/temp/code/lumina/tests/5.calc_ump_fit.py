import pdb
from ultron.ump.core import env
from ultron.ump.trade.ml_feature import append_user_feature
from ultron.ump.core.ump import run_loop_back

from lumina import env as lumina_env
from lumina.factors.buy import MinFactorChaosBuyL,MinFactorBollBandtBuyL
from lumina.factors.sell import MinFactorAtrNStop
from lumina.position import KellyPosition
from lumina.agent import main_lbm

from create_data import create_cnstock_box, create_agent_data

lumina_env.g_max_window = 3
env.g_market_target = env.EMarketTargetType.E_MARKET_TARGET_CN
env.g_enable_ml_feature = True
for featue in main_lbm.g_feature_list:
    append_user_feature(featue)


def fit():
    buy_factors = [{
        'class': MinFactorChaosBuyL,
        'atr_xd': 50,
        'position': {
            'class': KellyPosition
        }
    },{
        'class': MinFactorBollBandtBuyL,
        'ma_xd': 20,
        'position': {
            'class': KellyPosition
        }
    }]

    sell_factors = [{
        'class': MinFactorAtrNStop,
        'stop_loss_n': 0.5,
        'stop_win_n': 2.0
    }]

    read_cash = 5000000
    benchmark_kl_pd, pick_kl_pd_dict, choice_symbols = create_cnstock_box()

    result_tuple, kl_pd_manager = run_loop_back(
        read_cash=read_cash,
        buy_factors=buy_factors,
        sell_factors=sell_factors,
        benchmark_kl_pd=benchmark_kl_pd,
        pick_kl_pd_dict=pick_kl_pd_dict,
        choice_symbols=choice_symbols)
    orders_pd_train = result_tuple.orders_pd
    pdb.set_trace()

    ## 训练模型
    main_lbm.MainLBM.agent_main_clf_dump(
        orders_pd_train,
        market_name=env.EMarketTargetType.E_MARKET_TARGET_CN,
        threshold=0.5)


def predict():
    agent = main_lbm.MainLBM(
        market_name=env.EMarketTargetType.E_MARKET_TARGET_CN, predict=True)
    factors_data = create_agent_data(ticker_dim=1,
                                     columns=agent.get_predict_col())
    result = agent.predict(factors_data.values.reshape(1, -1))


fit()
