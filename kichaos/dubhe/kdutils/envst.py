import pdb
#from envs.trader import LongAlphaTraderEnv
from kichaos.envs.trader.cn_ashares import Long004TraderEnv as LongTraderEnv


def create_env(id, initial_amount, data, liaison_id, liaison_window, envory_id,
               envory_window, state_transformer_class, actor_transformer_class,
               critic_transformer_class, buy_cost_pct, sell_cost_pct, step_len,
               features, targets, mode, variant):
    #nxt1_columns = data.filter(regex="^nxt1_").columns.to_list()

    ## 特征周期滚动算
    #features = [
    #    col for col in data.columns
    #    if col not in ['trade_date', 'code'] + nxt1_columns
    #]
    ### 原始因子特征
    ticker_dimension = len(data.code.unique())
    state_space = ticker_dimension
    env = LongTraderEnv(id=id,
                           liaison_id=liaison_id,
                           liaison_window=liaison_window,
                           envory_id=envory_id,
                           envory_window=envory_window,
                           df=data,
                           features=features,
                           targets=targets,
                           state_space=state_space,
                           action_dim=ticker_dimension,
                           initial_amount=initial_amount,
                           buy_cost_pct=[buy_cost_pct] * ticker_dimension,
                           sell_cost_pct=[sell_cost_pct] * ticker_dimension,
                           ticker_dim=ticker_dimension,
                           cont_multnum={},
                           mode=mode,
                           step_len=step_len,
                           state_transformer_class=state_transformer_class,
                           actor_transformer_class=actor_transformer_class,
                           critic_transformer_class=critic_transformer_class,
                           **variant)
    return env
