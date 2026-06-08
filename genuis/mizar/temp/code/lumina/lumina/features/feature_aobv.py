# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_aobv


class FeatureAobv(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.aobv_keys = frozenset([(4, 12, 1), (8, 24, 1),
                                    (4, 12, 0)])  #fast, slow, ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}volume_aobv{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], w) for dk in self.aobv_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      aobv_dict):
        for dk in self.aobv_keys:
            day_th = dk[1] * 2 + 1
            if day_ind - day_th - window >= 0:
                aobv_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                aobv_df = combine_kl_pd[-day_th -
                                        window:] if combine_kl_pd.shape[0] > (
                                            day_th + window) else combine_kl_pd
            aobv = calc_aobv(aobv_df,
                             fast=dk[0],
                             slow=dk[1],
                             ewm=True if dk[2] == 1 else False)
            aobv_score = aobv.close
            aobv_score = 0 if np.isnan(aobv_score) else round(
                aobv_score, g_point_num)
            aobv_dict['{}volume_aobv{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], window)] = aobv_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        aobv_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               aobv_dict=aobv_dict)
        return aobv_dict
