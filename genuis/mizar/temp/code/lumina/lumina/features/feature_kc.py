# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_kc


class FeatureKC(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.kc_keys = frozenset([(20, 2, 1), (20, 2, 0)])
        self.kc_names = frozenset(['upper', 'lower'])
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}kc{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], n, w) for dk in self.kc_keys for n in self.kc_names
            for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      kc_dict):
        for dk in self.kc_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                kc_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                              window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                kc_df = combine_kl_pd[-day_th -
                                      window:] if combine_kl_pd.shape[0] > (
                                          day_th + window) else combine_kl_pd
            upper, lower = calc_kc(kc_df,
                                   xd=dk[0],
                                   scalar=dk[1],
                                   ewm=True if dk[2] == 1 else False)

            upper_score = upper.close
            lower_score = lower.close

            upper_score = 0 if np.isnan(upper_score) else round(
                upper_score, g_point_num)
            lower_score = 0 if np.isnan(lower_score) else round(
                lower_score, g_point_num)

            kc_dict['{}kc{}_{}_{}_upper_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], window)] = upper_score

            kc_dict['{}kc{}_{}_{}_lower_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], window)] = lower_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        kc_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               kc_dict=kc_dict)
        return kc_dict
