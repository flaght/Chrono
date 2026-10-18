# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_thick


class FeatureThick(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.thick_keys = frozenset([
            (10, 0.1, 1, 1), (15, 0.1, 1, 0), (20, 0.9, 0, 1), (20, 0.9, 0, 0)
        ])  # xd, quant=0.1, direction=True, ewm=True
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_thick{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0],
                int(dk[1] * 10), dk[2], dk[3], w) for dk in self.thick_keys
            for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      thick_dict):
        for dk in self.thick_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                thick_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                 window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                thick_df = combine_kl_pd[-day_th -
                                         window:] if combine_kl_pd.shape[0] > (
                                             day_th +
                                             window) else combine_kl_pd
            thick = calc_thick(thick_df,
                               xd=dk[0],
                               quant=dk[1],
                               direction=True if dk[2] == 1 else False,
                               ewm=True if dk[3] == 1 else False)
            thick_score = thick.close
            thick_score = 0 if np.isnan(thick_score) else round(
                thick_score, g_point_num)
            thick_dict['{}price_thick{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0],
                int(dk[1] * 10), dk[2], dk[3], window)] = thick_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        thick_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               thick_dict=thick_dict)
        return thick_dict
