# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_stochrsi


class FeatureStochRSI(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.stochrsi_keys = frozenset([(10, 10, 5, 1, 1), (10, 10, 5, 1, 0)
                                        ])  # xd, rsi_xd, fast_xd, scalar,ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_stochrsi{}_{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], dk[4], w) for dk in self.stochrsi_keys
            for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      stochrsi_dict):
        for dk in self.stochrsi_keys:
            day_th = dk[0] * 3 + 1
            if day_ind - day_th - window >= 0:
                stochrsi_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                    window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                stochrsi_df = combine_kl_pd[
                    -day_th - window:] if combine_kl_pd.shape[0] > (
                        day_th + window) else combine_kl_pd
            stochrsi = calc_stochrsi(stochrsi_df,
                                     xd=dk[0],
                                     rsi_xd=dk[1],
                                     fast_xd=dk[2],
                                     scalar=dk[3],
                                     ewm=True if dk[4] == 1 else False)
            stochrsi_score = stochrsi.close
            stochrsi_score = 0 if np.isnan(stochrsi_score) else round(
                stochrsi_score, g_point_num)
            stochrsi_dict['{}price_stochrsi{}_{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], dk[4], window)] = stochrsi_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        stochrsi_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               stochrsi_dict=stochrsi_dict)
        return stochrsi_dict
