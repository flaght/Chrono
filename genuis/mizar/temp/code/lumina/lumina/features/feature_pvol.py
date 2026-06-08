# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_pvol


class FeaturePVOL(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.pvol_keys = frozenset([2])  # offset
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_pvol{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk, w)
            for dk in self.pvol_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      pvol_dict):
        for dk in self.pvol_keys:
            day_th = 1
            if day_ind - day_th - window >= 0:
                pvol_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                pvol_df = combine_kl_pd[-day_th -
                                        window:] if combine_kl_pd.shape[0] > (
                                            day_th + window) else combine_kl_pd

            pvol = calc_pvol(pvol_df)
            pvol_score = pvol.close
            pvol_score = 0 if np.isnan(pvol_score) else round(
                pvol_score, g_point_num)
            pvol_dict['{}price_pvol{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk,
                window)] = pvol_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        pvol_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               pvol_dict=pvol_dict)
        return pvol_dict
