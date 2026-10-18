# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_clkbar


class FeatureCLKBAR(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.clkbar_keys = frozenset([(7, 1), (14, 1)])  # xd, ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_clkbar{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1], w)
            for dk in self.clkbar_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      clkbar_dict):
        for dk in self.clkbar_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                clkbar_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                  window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                clkbar_df = combine_kl_pd[-day_th -
                                          window:] if combine_kl_pd.shape[
                                              0] > (day_th +
                                                    window) else combine_kl_pd
            clkbar = calc_clkbar(clkbar_df,
                                 xd=dk[0],
                                 ewm=True if dk[1] == 1 else False)
            clkbar_score = clkbar.close
            clkbar_score = 0 if np.isnan(clkbar_score) else round(
                clkbar_score, g_point_num)
            clkbar_dict['{}price_clkbar{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                window)] = clkbar_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        clkbar_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               clkbar_dict=clkbar_dict)
        return clkbar_dict
