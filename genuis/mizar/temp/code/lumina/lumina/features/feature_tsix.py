# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_tsix


class FeatureTSIX(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.tsix_keys = frozenset([(5, 1, 2, 1), (10, 1, 2, 0),
                                    (5, 1, 2, 0)])  # xd, scalar, drift ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_tsix{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], w) for dk in self.tsix_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      tsix_dict):
        for dk in self.tsix_keys:
            day_th = dk[0] * 3 + 1
            if day_ind - day_th - window >= 0:
                tsix_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                tsix_df = combine_kl_pd[-day_th -
                                        window:] if combine_kl_pd.shape[0] > (
                                            day_th + window) else combine_kl_pd
            tsix = calc_tsix(tsix_df,
                             xd=dk[0],
                             scalar=dk[1],
                             drift=dk[2],
                             ewm=True if dk[3] == 1 else False)
            tsix_score = tsix.close
            tsix_score = 0 if np.isnan(tsix_score) else round(
                tsix_score, g_point_num)
            tsix_dict['{}price_tsix{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], window)] = tsix_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        tsix_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               tsix_dict=tsix_dict)
        return tsix_dict
