# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_eom


class FeatureEOM(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.eom_keys = frozenset([(14, 1, 0, 1), (7, 1, 0, 1),
                                   (14, 1, 0, 0)])  # xd drift divisor ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}volume_eom{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], w) for dk in self.eom_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      eom_dict):
        for dk in self.eom_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                eom_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                eom_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd
            eom = calc_eom(eom_df,
                           xd=dk[0],
                           drift=dk[1],
                           divisor=dk[2],
                           ewm=True if dk[3] == 1 else False)
            eom_score = eom.close
            eom_score = 0 if np.isnan(eom_score) else round(
                eom_score, g_point_num)
            eom_dict['{}volume_eom{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], window)] = eom_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        eom_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               eom_dict=eom_dict)
        return eom_dict
