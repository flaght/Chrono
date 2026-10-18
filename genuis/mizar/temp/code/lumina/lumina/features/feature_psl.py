# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_psl


class FeaturePSL(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.psl_keys = frozenset([(7, 1), (14, 1), (7, 0)])  # xd, ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_psl{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1], w)
            for dk in self.psl_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      psl_dict):
        for dk in self.psl_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                psl_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                psl_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd
            psl = calc_psl(psl_df, xd=dk[0], ewm=True if dk[1] == 1 else False)
            psl_score = psl.close
            psl_score = 0 if np.isnan(psl_score) else round(
                psl_score, g_point_num)
            psl_dict['{}price_psl{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                window)] = psl_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        psl_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               psl_dict=psl_dict)
        return psl_dict
