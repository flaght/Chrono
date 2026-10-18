# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_kvo


class FeatureKVO(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.kvo_keys = frozenset([(3, 10, 1, 1), (6, 15, 1, 1),
                                   (3, 10, 1, 0)])  # fast, slow, drift, ewn
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}volume_kvo{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], w) for dk in self.kvo_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      kvo_dict):
        for dk in self.kvo_keys:
            day_th = dk[1] * 2 + 1
            if day_ind - day_th - window >= 0:
                kvo_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                kvo_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd
            kvo = calc_kvo(kvo_df,
                           fast=dk[0],
                           slow=dk[1],
                           drift=dk[2],
                           ewm=True if dk[3] == 1 else False)
            kvo_score = kvo.close
            kvo_score = 0 if np.isnan(kvo_score) else round(
                kvo_score, g_point_num)
            kvo_dict['{}volume_kvo{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], window)] = kvo_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        kvo_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               kvo_dict=kvo_dict)
        return kvo_dict
