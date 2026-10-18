# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_willr


class FeatureWillr(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.willr_keys = frozenset([5, 10, 7])  # xd
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_willr{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk, w)
            for dk in self.willr_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      willr_dict):
        for dk in self.willr_keys:
            day_th = dk
            if day_ind - day_th - window >= 0:
                willr_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                 window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                willr_df = combine_kl_pd[-day_th -
                                         window:] if combine_kl_pd.shape[0] > (
                                             day_th +
                                             window) else combine_kl_pd
            willr = calc_willr(willr_df, xd=dk)
            willr_score = willr.close
            willr_score = 0 if np.isnan(willr_score) else round(
                willr_score, g_point_num)
            willr_dict['{}price_willr{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk,
                window)] = willr_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        willr_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               willr_dict=willr_dict)
        return willr_dict
