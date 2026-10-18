# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_dema


class FeatureDema(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.dema_keys = frozenset([(10, 1), (20, 1), (10, 0)])  # xd, ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}dema{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1], w)
            for dk in self.dema_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      dema_dict):
        for dk in self.dema_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                dema_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                dema_df = combine_kl_pd[-day_th -
                                        window:] if combine_kl_pd.shape[0] > (
                                            day_th + window) else combine_kl_pd
            dema = calc_dema(dema_df,
                             xd=dk[0],
                             ewm=True if dk[1] == 1 else False)
            dema_score = dema.close
            dema_score = 0 if np.isnan(dema_score) else round(
                dema_score, g_point_num)
            dema_dict['{}dema{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                window)] = dema_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        dema_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               dema_dict=dema_dict)
        return dema_dict
