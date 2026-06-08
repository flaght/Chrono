# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_t3


class FeatureT3(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.t3_keys = frozenset([(10, 1), (20, 1), (10, 0)])  # xd , ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}t3{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1], w)
            for dk in self.t3_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      t3_dict):
        for dk in self.t3_keys:
            day_th = dk[0] * 3 + 1
            if day_ind - day_th - window >= 0:
                t3_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                              window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                t3_df = combine_kl_pd[-day_th -
                                      window:] if combine_kl_pd.shape[0] > (
                                          day_th + window) else combine_kl_pd
            t3 = calc_t3(t3_df, xd=dk[0], ewm=True if dk[1] == 1 else False)
            t3_score = t3.close
            t3_score = 0 if np.isnan(t3_score) else round(
                t3_score, g_point_num)
            t3_dict['{}t3{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                window)] = t3_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        t3_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               t3_dict=t3_dict)
        return t3_dict
