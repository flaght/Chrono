# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_tsi


class FeatureTSI(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.tsi_keys = frozenset([(10, 5, 5, 1, 1, 1), (5, 2, 5, 1, 1, 1)
                                   ])  # fast slow xd scalar drift ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_tsi{}_{}_{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], dk[4], dk[5], w) for dk in self.tsi_keys
            for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      tsi_dict):
        for dk in self.tsi_keys:
            day_th = dk[0] * 3 + 1
            if day_ind - day_th - window >= 0:
                tsi_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                tsi_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd
            tsi = calc_tsi(tsi_df,
                           fast=dk[0],
                           slow=dk[1],
                           xd=dk[2],
                           scalar=dk[3],
                           drift=dk[4],
                           ewm=True if dk[5] == 1 else False)
            tsi_score = tsi.close
            tsi_score = 0 if np.isnan(tsi_score) else round(
                tsi_score, g_point_num)
            tsi_dict['{}price_tsi{}_{}_{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], dk[4], dk[5], window)] = tsi_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        tsi_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               tsi_dict=tsi_dict)
        return tsi_dict
