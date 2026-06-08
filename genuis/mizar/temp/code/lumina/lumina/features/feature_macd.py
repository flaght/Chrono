# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_macd


class FeatureMACD(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.macd_keys = frozenset([(5, 10, 1), (7, 14, 1),
                                    (5, 10, 0)])  # fast, slow, ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_macd{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], w) for dk in self.macd_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      macd_dict):
        for dk in self.macd_keys:
            day_th = dk[1] * 2 + 1
            if day_ind - day_th - window >= 0:
                macd_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                macd_df = combine_kl_pd[-day_th -
                                        window:] if combine_kl_pd.shape[0] > (
                                            day_th + window) else combine_kl_pd
            macd = calc_macd(macd_df,
                             fast=dk[0],
                             slow=dk[1],
                             ewm=True if dk[2] == 1 else False)
            macd_score = macd.close
            macd_score = 0 if np.isnan(macd_score) else round(
                macd_score, g_point_num)
            macd_dict['{}price_macd{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], window)] = macd_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        macd_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               macd_dict=macd_dict)
        return macd_dict
