# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_nvi


class FeatureNVI(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.nvi_keys = frozenset([(10, 100), (15, 100),
                                   (20, 100)])  # xd  scalar
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}volume_nvi{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1], w)
            for dk in self.nvi_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      nvi_dict):
        for dk in self.nvi_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                nvi_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                nvi_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd
            nvi = calc_nvi(nvi_df, xd=dk[0], scalar=dk[1])
            nvi_score = nvi.close
            nvi_score = 0 if np.isnan(nvi_score) else round(
                nvi_score, g_point_num)
            nvi_dict['{}volume_nvi{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                window)] = nvi_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        nvi_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               nvi_dict=nvi_dict)
        return nvi_dict
