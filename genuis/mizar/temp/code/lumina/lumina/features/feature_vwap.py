# -*- encoding:utf-8 -*-
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical.vwap import calc_vwap

class FeatureVWap(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.vwap_keys = frozenset([21, 42, 60])
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}vwap_ang{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk, w)
            for dk in self.vwap_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      vwap_dict):
        for dk in self.vwap_keys:
            if day_ind - dk - window >= 0:
                vwap_close = kl_pd[day_ind - dk + 1 - window:day_ind + 1 -
                                   window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                vwap_close = combine_kl_pd[-dk -
                                           window:] if combine_kl_pd.shape[
                                               0] > (dk +
                                                     window) else combine_kl_pd
            ##
            vwap_price = calc_vwap(vwap_close)
            vwap_score = vwap_price.score
            vwap_score = 0 if np.isnan(vwap_score) else round(
                vwap_score, g_point_num)
            vwap_dict['{}vwap_ang{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk,
                window)] = vwap_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        vwap_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               vwap_dict=vwap_dict)
        return vwap_dict
