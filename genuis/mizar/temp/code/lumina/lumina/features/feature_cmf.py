# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_cmf


class FeatureCMF(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.cmf_keys = frozenset([(10, 1), (15, 1), (20, 0)])  # xd ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}volume_cmf{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1], w)
            for dk in self.cmf_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      cmf_dict):
        for dk in self.cmf_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                cmf_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                cmf_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd
            cmf = calc_cmf(cmf_df, xd=dk[0], ewm=True if dk[1] == 1 else False)
            cmf_score = cmf.close
            cmf_score = 0 if np.isnan(cmf_score) else round(
                cmf_score, g_point_num)
            cmf_dict['{}volume_cmf{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                window)] = cmf_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        cmf_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               cmf_dict=cmf_dict)
        return cmf_dict
