# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_vwma


class FeatureVWMA(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.vwma_keys = frozenset([(3, 1), (6, 1), (3, 0)])  # xd, ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}vwma{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1], w)
            for dk in self.vwma_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      vwma_dict):
        for dk in self.vwma_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                vwma_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                vwma_df = combine_kl_pd[-day_th -
                                        window:] if combine_kl_pd.shape[0] > (
                                            day_th + window) else combine_kl_pd
            vwma = calc_vwma(vwma_df,
                             xd=dk[0],
                             ewm=True if dk[1] == 1 else False)
            vwma_score = vwma.close
            vwma_score = 0 if np.isnan(vwma_score) else round(
                vwma_score, g_point_num)
            vwma_dict['{}vwma{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                window)] = vwma_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        vwma_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               vwma_dict=vwma_dict)
        return vwma_dict
