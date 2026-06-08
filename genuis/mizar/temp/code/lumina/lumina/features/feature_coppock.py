# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_coppock


class FeatureCOPPOCK(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.coppock_keys = frozenset([(5, 10, 14, 1.0, 1, 1),
                                       (7, 14, 21, 1.0, 1, 1),
                                       (5, 10, 14, 1.0, 1, 0)
                                       ])  # fast slow xd scalar drift
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_coppock{}_{}_{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], dk[4], dk[5], w) for dk in self.coppock_keys
            for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      coppock_dict):
        for dk in self.coppock_keys:
            day_th = dk[1] * 3 + 1
            if day_ind - day_th - window >= 0:
                coppock_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                   window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                coppock_df = combine_kl_pd[-day_th -
                                           window:] if combine_kl_pd.shape[
                                               0] > (day_th +
                                                     window) else combine_kl_pd
            coppock = calc_coppock(coppock_df,
                                   fast=dk[0],
                                   slow=dk[1],
                                   xd=dk[2],
                                   scalar=dk[3],
                                   drift=dk[4],
                                   ewm=True if dk[5] == 1 else False)
            coppock_score = coppock.close
            coppock_score = 0 if np.isnan(coppock_score) else round(
                coppock_score, g_point_num)
            coppock_dict['{}price_coppock{}_{}_{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], dk[4], dk[5], window)] = coppock_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        coppock_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               coppock_dict=coppock_dict)
        return coppock_dict
