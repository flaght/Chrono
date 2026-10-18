# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_volatility2


class FeatureVolatility2(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.volatility2_keys = frozenset([
            (5, 30, 2, 7, 1), (10, 60, 5, 15, 1), (5, 30, 2, 7, 1)
        ])  # fast low fdrift sdrift ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}volatility2{}_{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], dk[4], w) for dk in self.volatility2_keys
            for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      volatility2_dict):
        for dk in self.volatility2_keys:
            day_th = dk[1] * 2 + 1
            if day_ind - day_th - window >= 0:
                volatility2_df = kl_pd[day_ind - day_th + 1 - window:day_ind +
                                       1 - window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                volatility2_df = combine_kl_pd[
                    -day_th - window:] if combine_kl_pd.shape[0] > (
                        day_th + window) else combine_kl_pd
            volatility2 = calc_volatility2(volatility2_df,
                                           fast=dk[0],
                                           low=dk[1],
                                           fdrift=dk[2],
                                           sdrift=dk[3],
                                           ewm=True if dk[4] == 1 else False)
            volatility2_score = volatility2.close
            volatility2_score = 0 if np.isnan(volatility2_score) else round(
                volatility2_score, g_point_num)
            volatility2_dict['{}volatility2{}_{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], dk[4], window)] = volatility2_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        volatility2_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               volatility2_dict=volatility2_dict)
        return volatility2_dict
