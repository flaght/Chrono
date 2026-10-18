# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_brar


class FeatureBRAR(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.brar_keys = frozenset([(5, 1, 2, 1), (10, 1, 2, 1),
                                    (5, 1, 2, 0)])  # xd, scalar drift ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_brar{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], w) for dk in self.brar_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      brar_dict):
        for dk in self.brar_keys:
            day_th = dk[1] * 2 + 1
            if day_ind - day_th - window >= 0:
                brar_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                brar_df = combine_kl_pd[-day_th -
                                        window:] if combine_kl_pd.shape[0] > (
                                            day_th + window) else combine_kl_pd
            brar = calc_brar(brar_df,
                             xd=dk[0],
                             drift=dk[2],
                             ewm=True if dk[3] == 1 else False)
            brar_score = brar.close
            brar_score = 0 if np.isnan(brar_score) else round(
                brar_score, g_point_num)
            brar_dict['{}price_brar{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], window)] = brar_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        brar_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               brar_dict=brar_dict)
        return brar_dict
