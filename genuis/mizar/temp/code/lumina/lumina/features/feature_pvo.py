# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_pvo


class FeaturePVO(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.pvo_keys = frozenset([(7, 14, 1, 1), (5, 10, 1, 0),
                                   (5, 10, 1, 1)])  # fast slow, scalar, ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_pvo{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], w) for dk in self.pvo_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      pvo_dict):
        for dk in self.pvo_keys:
            day_th = dk[1] * 2 + 1
            if day_ind - day_th - window >= 0:
                pvo_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                pvo_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd
            pvo = calc_pvo(pvo_df,
                           fast=dk[0],
                           slow=dk[1],
                           scalar=dk[2],
                           ewm=True if dk[3] == 1 else False)
            pvo_score = pvo.close
            pvo_score = 0 if np.isnan(pvo_score) else round(
                pvo_score, g_point_num)
            pvo_dict['{}price_pvo{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], window)] = pvo_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        pvo_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               pvo_dict=pvo_dict)
        return pvo_dict
