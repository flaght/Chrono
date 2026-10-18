# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_pvi


class FeaturePVI(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.pvi_keys = frozenset([(10, 100), (15, 100),
                                   (20, 100)])  # xd  scalar
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_pvi{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1], w)
            for dk in self.pvi_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      pvi_dict):
        for dk in self.pvi_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                pvi_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                pvi_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd
            pvi = calc_pvi(pvi_df, xd=dk[0], scalar=dk[1])
            pvi_score = pvi.close
            pvi_score = 0 if np.isnan(pvi_score) else round(
                pvi_score, g_point_num)
            pvi_dict['{}price_pvi{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                window)] = pvi_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        pvi_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               pvi_dict=pvi_dict)
        return pvi_dict
