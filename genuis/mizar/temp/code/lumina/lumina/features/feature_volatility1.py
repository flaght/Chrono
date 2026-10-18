# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_volatility1


class FeatureVolatility1(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.volatility1_keys = frozenset([(10, 1, 1), (20, 1, 1),
                                           (10, 1, 0)])  # xd , drift, ewm
        self.volatility1_name = frozenset(['boll_cls2up', 'boll_cls2dow'])
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}volatility1{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], n, w) for dk in self.volatility1_keys
            for n in self.volatility1_name for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      volatility1_dict):
        for dk in self.volatility1_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                volatility1_df = kl_pd[day_ind - day_th + 1 - window:day_ind +
                                       1 - window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                volatility1_df = combine_kl_pd[
                    -day_th - window:] if combine_kl_pd.shape[0] > (
                        day_th + window) else combine_kl_pd
            boll_cls2up, boll_cls2dow = calc_volatility1(
                volatility1_df,
                xd=dk[0],
                drift=dk[1],
                ewm=True if dk[2] == 1 else False)
            boll_cls2up_score = boll_cls2up.close
            boll_cls2dow_score = boll_cls2dow.close

            boll_cls2up_score = 0 if np.isnan(boll_cls2up_score) else round(
                boll_cls2up_score, g_point_num)
            boll_cls2dow_score = 0 if np.isnan(boll_cls2dow_score) else round(
                boll_cls2dow_score, g_point_num)

            volatility1_dict['{}volatility1{}_{}_{}_boll_cls2up_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], window)] = boll_cls2up_score

            volatility1_dict['{}volatility1{}_{}_{}_boll_cls2dow_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], window)] = boll_cls2dow_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        volatility1_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               volatility1_dict=volatility1_dict)
        return volatility1_dict
