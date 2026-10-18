# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical.atr import calc_atr
#from ultron.ump.technical.atr import calc_atr_std


class FeatureAtr(FeatureBase, BuyFeatureMixin):

    def __init__(self):
        self.atr_keys = frozenset([(7, 1), (14, 0), (21, 1)])  # xd, ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}atr{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1], w)
            for dk in self.atr_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      atr_dict):
        for dk in self.atr_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                atr_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                atr_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd
            atr = calc_atr(atr_df,
                               xd=dk[0],
                               ewm=True if dk[1] == 1 else False)
            atr_score = atr.score
            atr_score = 0 if np.isnan(atr_score) else round(
                atr_score, g_point_num)
            atr_dict['{}atr{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                window)] = atr_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        atr_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               atr_dict=atr_dict)
        return atr_dict
