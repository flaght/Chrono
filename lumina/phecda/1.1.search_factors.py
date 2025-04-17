import os, pdb
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from kdutils.factors import factors_sets
from kdutils.macro import base_path, codes
from ultron.kdutils.file import load_pickle,dump_pickle
from ultron.kdutils import date as date_utils
from ultron.env import *
from ultron.ump.core import env
from ultron.ump.metrics import grid_helper as GridHelper
from ultron.ump.metrics.grid_search import GridSearch
from ultron.ump.metrics.score import BaseScorer
from ultron.ump.metrics.score import make_scorer

weights = [
    0,  # all_profit 
    1,  # profit_rate,
    1,  # win_rate,
    1,  # gains
    1,  # losses,
    1,  # profit_win,
    1,  # profit_loss,
    0,  # trader_count_mean,
    0,  # trader_count_max,
    0,  # trader_count_min,
    0,  # deal_count_mean,
    0,  # deal_count_max,
    0,  # deal_count_min,
    0,  # keep_count_mean,
    0,  # keep_count_max,
    0  # keep_count_min
]



def create_buy_factors(name):
    factor_grid = factors_sets[name]['buy']
    buy_factors_product = GridHelper.gen_factor_grid(
        GridHelper.K_GEN_FACTOR_PARAMS_BUY, factor_grid)
    return buy_factors_product


def create_sell_factors(name):
    factor_grid = factors_sets[name]['sell']
    factor_risk = factors_sets[name]['risk']

    sell_factors_product = GridHelper.gen_factor_grid(
        GridHelper.K_GEN_FACTOR_PARAMS_SELL,
        [factor_risk,factor_grid],
        need_empty_sell=False)
    return sell_factors_product


def load_data(method):
    dirs = os.path.join(base_path, method)
    benchmark_kl_pd = load_pickle(
        os.path.join(dirs,
                     'benckmark_{0}.pkl'.format(os.environ['INSTRUMENTS'])))
    pick_kl_pd_dict = load_pickle(
        os.path.join(dirs, 'pick_{0}.pkl'.format(os.environ['INSTRUMENTS'])))
    '''
    factors_data = pd.read_feather(
        '/workspace/data/dev/kd/evolution/nn/phecda/nicso/rbb/envoy008_5_1_1_0nw_1nc_11.feather'
    )
    for k, v in pick_kl_pd_dict.items():
        factors_data = factors_data.set_index('trade_time')
        factors_data['code'] = factors_data['code'] + '0'
        factors_data.index.name = 'trade_date'
        factors_data.index = pd.to_datetime(factors_data.index)
        v.index = pd.to_datetime(v.index)
        #factors_data = factors_data.reset_index()
        #v = v.reset_index()
        v = pd.concat([v, factors_data], axis=1).dropna(subset=['rbb', 'low'])
        v.rename(columns={'rbb': 'pred'}, inplace=True)
        pdb.set_trace()
        pick_kl_pd_dict[k] = v
    '''
    choice_symbols = load_pickle(
        os.path.join(dirs, 'choice_{0}.pkl'.format(os.environ['INSTRUMENTS'])))
    return benchmark_kl_pd, pick_kl_pd_dict, choice_symbols


### 胜率 卡玛 盈亏比
class PhecdaScorer(BaseScorer):

    def _init_self_begin(self, *arg, **kwargs):
        """胜率，卡玛，盈亏比，收益，策略最大回撤组成select_score_func"""
        self.select_score_func = lambda metrics: [
            metrics.win_rate, 0 if np.isnan(metrics.algorithm_calmar) else
            metrics.algorithm_calmar, metrics.win_loss_profit_rate, metrics.
            algorithm_period_returns, metrics.max_drawdown
        ]
        self.columns_name = [
            'win_rate', 'calmar', 'wlpr', 'returns', 'max_drawdown'
        ]
        self.weights_cnt = len(self.columns_name)

    def _init_self_end(self, *arg, **kwargs):
        pass


class OhwlScorer(BaseScorer):

    def _init_self_begin(self, *arg, **kwargs):
        ### all_profit 策略总收益
        ### win_rate 胜率
        ### win_loss_profit_rate 盈亏比
        ### gains_mean 策略期望收益 平均每笔交易的盈利百分比
        ### losses_mean 策略期望亏损 平均每笔交易的亏损百分比
        ### profit_cg_win_sum  每一笔交易使用相同的资金，策略的总获利交易获利比例和
        ### profit_cg_loss_sum 每一笔交易使用相同的资金，策略的总亏损交易亏损比例和

        self.select_score_func = lambda metrics: [
            metrics.all_profit, metrics.win_loss_profit_rate, metrics.win_rate,
            metrics.gains_mean, metrics.losses_mean, metrics.profit_cg_win_sum,
            metrics.profit_cg_loss_sum, metrics.trader_days, metrics.
            trader_count_mean, metrics.trader_count_max, metrics.
            trader_count_min, metrics.deal_count_mean, metrics.deal_count_max,
            metrics.deal_count_min, metrics.keep_count_mean, metrics.
            keep_count_max, metrics.keep_count_min
        ]
        self.columns_name = [
            'all_profit', 'profit_rate', 'win_rate', 'gains', 'losses',
            'profit_win', 'profit_loss', 'trader_days', 'trader_count_mean',
            'trader_count_max', 'trader_count_min', 'deal_count_mean',
            'deal_count_max', 'deal_count_min', 'keep_count_mean',
            'keep_count_max', 'keep_count_min'
        ]
        self.weights_cnt = len(self.columns_name)

    def _init_self_end(self, *arg, **kwargs):
        pass


def train(method):
    benchmark_kl_pd, pick_kl_pd_dict, choice_symbols = load_data(method)
    benchmark_kl_pd.name = 'IF0'
    sell_factors_product = create_sell_factors(name=os.environ['INSTRUMENTS'])
    buy_factors_product = create_buy_factors(name=os.environ['INSTRUMENTS'])
    pdb.set_trace()
    print('卖出因子参数共有{}种组合方式'.format(len(sell_factors_product)))
    #print('卖出因子组合0: 形式为{}'.format(sell_factors_product[0]))
    print('买入因子参数共有{}种组合方式'.format(len(buy_factors_product)))
    #print('买入因子组合形式为{}'.format(buy_factors_product))
    print('组合因子参数数量{}'.format(
        len(buy_factors_product) * len(sell_factors_product)))
    pdb.set_trace()
    env.g_market_target = env.EMarketTargetType.E_MARKET_TARGET_FUTURES_CN
    env.g_enable_ml_feature = False
    read_cash = 50000000
    grid_search = GridSearch(read_cash,
                             choice_symbols,
                             benchmark_kl_pd=benchmark_kl_pd,
                             buy_factors_product=buy_factors_product,
                             sell_factors_product=sell_factors_product)

    grid_search.kl_pd_manager.set_pick_time(pick_kl_pd_dict)
    pdb.set_trace()
    score_tuple_array = grid_search.train(n_jobs=64)
    pdb.set_trace()
    print('score_tuple_array', score_tuple_array)


def makescore(method):
    pdb.set_trace()
    score_tuple_array = load_pickle('rbb_score_tuple_array.pkl')
    #for score_tuple in score_tuple_array:
    #    print("buy:{0}".format(score_tuple.buy_factors))
    #    print("sell:{0}".format(score_tuple.sell_factors))

    #    print('--->')
    #pdb.set_trace()
    scorer = OhwlScorer(score_tuple_array=score_tuple_array, weights=weights)
    pdb.set_trace()
    print


def make1():
    orders_pd = pd.read_feather('111.feather').set_index('index')
    pdb.set_trace()
    orders_pd['profit_cg'] = orders_pd['profit'] / (orders_pd['buy_price'] *
                                                    orders_pd['buy_cnt'])
    deal_pd = orders_pd[orders_pd['sell_type'].isin(['win', 'loss'])]
    dumm_sell = pd.get_dummies(deal_pd.sell_type_extra)
    dumm_sell_t = dumm_sell.T
    dumm_sell_t_sum = dumm_sell_t.sum(axis=1)

    dumm_buy = pd.get_dummies(deal_pd.buy_factor)
    dumm_buy = dumm_buy.T
    dumm_buy_t_sum = dumm_buy.sum(axis=1)

    orders_pd['buy_date'] = orders_pd['buy_date'].astype(int)
    orders_pd[orders_pd['result'] != 0]['sell_date'].astype(int, copy=False)
    orders_pd['keep_days'] = orders_pd.apply(lambda x: date_utils.diff(
        x['buy_date'],
        date_utils.current_date_int() if x['result'] == 0 else x['sell_date']),
                                             axis=1)
    order_has_ret = orders_pd[orders_pd['result'] != 0]
    # 筛出未成交的单子
    order_keep = orders_pd[orders_pd['result'] == 0]

    xt = order_has_ret.result.value_counts()
    # 计算胜率
    if xt.shape[0] == 2:
        win_rate = xt[1] / xt.sum()
    elif xt.shape[0] == 1:
        win_rate = xt.index[0]
    else:
        win_rate = 0
    win_rate = win_rate
    # 策略持股天数平均值
    keep_days_mean = orders_pd['keep_days'].mean()
    # 策略持股天数中位数
    keep_days_median = orders_pd['keep_days'].median()

    # 策略期望收益
    gains_mean = order_has_ret[order_has_ret['profit_cg'] > 0].profit_cg.mean()
    if np.isnan(gains_mean):
        gains_mean = 0.0
    # 策略期望亏损
    losses_mean = order_has_ret[order_has_ret['profit_cg'] <
                                0].profit_cg.mean()
    if np.isnan(losses_mean):
        losses_mean = 0.0

        # 忽略仓位控的前提下，即假设每一笔交易使用相同的资金，策略的总获利交易获利比例和
    profit_cg_win_sum = order_has_ret[order_has_ret['profit_cg'] >
                                      0].profit.sum()
    # 忽略仓位控的前提下，即假设每一笔交易使用相同的资金，策略的总亏损交易亏损比例和
    profit_cg_loss_sum = order_has_ret[order_has_ret['profit_cg'] <
                                       0].profit.sum()

    if profit_cg_win_sum * profit_cg_loss_sum == 0 and profit_cg_win_sum + profit_cg_loss_sum > 0:
        # 其中有一个是0的，要转换成一个最小统计单位计算盈亏比，否则不需要
        if profit_cg_win_sum == 0:
            profit_cg_win_sum = 0.01
        if profit_cg_loss_sum == 0:
            profit_cg_win_sum = 0.01
    profit_cg_win_sum = profit_cg_win_sum
    profit_cg_loss_sum = profit_cg_loss_sum
    #  忽略仓位控的前提下，计算盈亏比
    win_loss_profit_rate = 0 if profit_cg_loss_sum == 0 else -round(
        profit_cg_win_sum / profit_cg_loss_sum, 4)
    #  忽略仓位控的前提下，计算所有交易单的盈亏总会
    all_profit = order_has_ret['profit'].sum()
    pdb.set_trace()
    print()

#make1()
#train('exper')
makescore('exper')
