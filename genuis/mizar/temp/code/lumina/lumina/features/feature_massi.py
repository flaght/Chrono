# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_massi


class FeatureMassi(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.massi_keys = frozenset([(9, 25, 1),
                                     (9, 25, 0)])  # fast, slow, ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}massi{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], w) for dk in self.massi_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      massi_dict):
        for dk in self.massi_keys:
            day_th = dk[1] * 2 + 1
            if day_ind - day_th - window >= 0:
                massi_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                 window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                massi_df = combine_kl_pd[-day_th -
                                         window:] if combine_kl_pd.shape[0] > (
                                             day_th +
                                             window) else combine_kl_pd
            massi = calc_massi(massi_df,
                               fast=dk[0],
                               slow=dk[1],
                               ewm=True if dk[2] == 1 else False)
            massi_score = massi.close
            massi_score = 0 if np.isnan(massi_score) else round(
                massi_score, g_point_num)
            massi_dict['{}massi{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], window)] = massi_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        massi_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               massi_dict=massi_dict)
        return massi_dict
