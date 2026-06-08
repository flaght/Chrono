# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_bop


class FeatureBOP(FeatureBase, BuyFeatureMixin, SellFeatureMixin):
    """偏离特征，支持买入，卖出"""

    def __init__(self):
        self.bop_keys = frozenset([1])  # scalar
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_bop{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk, w)
            for dk in self.bop_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      bop_dict):
        for dk in self.bop_keys:
            day_th = 1
            if day_ind - day_th - window >= 0:
                bop_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                bop_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd

            bop = calc_bop(bop_df)
            bop_score = bop.close
            bop_score = 0 if np.isnan(bop_score) else round(
                bop_score, g_point_num)
            bop_dict['{}price_bop{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk,
                window)] = bop_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        bop_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               bop_dict=bop_dict)
        return bop_dict
