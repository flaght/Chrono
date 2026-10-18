# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_upchvolatility


class FeatureUPCHVolatility(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.upchvolatility_keys = frozenset([(5, 2, 1), (10, 2, 1),
                                              (5, 1, 0)])  #xd,drift,ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_upchvolatility{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], w) for dk in self.upchvolatility_keys
            for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      upchvolatility_dict):
        for dk in self.upchvolatility_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                upchvolatility_df = kl_pd[day_ind - day_th + 1 -
                                          window:day_ind + 1 - window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                upchvolatility_df = combine_kl_pd[
                    -day_th - window:] if combine_kl_pd.shape[0] > (
                        day_th + window) else combine_kl_pd
            upchvolatility = calc_upchvolatility(
                upchvolatility_df,
                xd=dk[0],
                drift=dk[1],
                ewm=True if dk[2] == 1 else False)
            upchvolatility_score = upchvolatility.close
            upchvolatility_score = 0 if np.isnan(
                upchvolatility_score) else round(upchvolatility_score,
                                                 g_point_num)
            upchvolatility_dict['{}price_upchvolatility{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], window)] = upchvolatility_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        upchvolatility_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               upchvolatility_dict=upchvolatility_dict)
        return upchvolatility_dict
