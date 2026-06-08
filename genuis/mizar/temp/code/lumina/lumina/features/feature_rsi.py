# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_rsi


class FeatureRSI(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.rsi_keys = frozenset([(5, 1, 1), (10, 1, 1),
                                   (5, 1, 0)])  # xd, scalar, ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_rsi{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], w) for dk in self.rsi_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      rsi_dict):
        for dk in self.rsi_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                rsi_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                rsi_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd
            rsi = calc_rsi(rsi_df,
                           xd=dk[0],
                           scalar=dk[1],
                           ewm=True if dk[2] == 1 else False)
            rsi_score = rsi.close
            rsi_score = 0 if np.isnan(rsi_score) else round(
                rsi_score, g_point_num)
            rsi_dict['{}price_rsi{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], window)] = rsi_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        rsi_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               rsi_dict=rsi_dict)
        return rsi_dict
