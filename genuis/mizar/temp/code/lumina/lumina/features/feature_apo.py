# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_apo


class FeatureAPO(FeatureBase, BuyFeatureMixin, SellFeatureMixin):
    """价格变动幅度特征，支持买入，卖出"""

    def __init__(self):
        self.apo_keys = frozenset([(5, 10, 1),
                                   (7, 14, 1)])  # (fast, slow, ewm)
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_apo{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], w) for dk in self.apo_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      apo_dict):
        for dk in self.apo_keys:
            day_th = dk[1] * 2 + 1
            if day_ind - day_th - window >= 0:
                apo_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                apo_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd

            apo = calc_apo(apo_df,
                           fast=dk[0],
                           slow=dk[1],
                           ewm=True if dk[2] == 1 else False)
            apo_score = apo.score
            apo_score = 0 if np.isnan(apo_score) else round(
                apo_score, g_point_num)
            apo_dict['{}price_apo{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], window)] = apo_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        apo_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               apo_dict=apo_dict)
        return apo_dict
