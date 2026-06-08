# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_minimum_sum


class FeatureMinimumSum(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.minimum_sum_keys = frozenset([(10, 1), (15, 1), (10, 0)])
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_minimum_sum{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1], w)
            for dk in self.minimum_sum_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      minimum_sum_dict):
        for dk in self.minimum_sum_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                minimum_sum_df = kl_pd[day_ind - day_th + 1 - window:day_ind +
                                       1 - window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                minimum_sum_df = combine_kl_pd[
                    -day_th - window:] if combine_kl_pd.shape[0] > (
                        day_th + window) else combine_kl_pd
            minimum_sum = calc_minimum_sum(minimum_sum_df,
                                           xd=dk[0],
                                           ewm=True if dk[1] == 1 else False)
            minimum_sum_score = minimum_sum.close
            minimum_sum_score = 0 if np.isnan(minimum_sum_score) else round(
                minimum_sum_score, g_point_num)
            minimum_sum_dict['{}price_minimum_sum{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                window)] = minimum_sum_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        minimum_sum_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               minimum_sum_dict=minimum_sum_dict)
        return minimum_sum_dict
