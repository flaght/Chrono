# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_volatility5


class FeatureVolatility5(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.volatility5_keys = frozenset([(10, 5, 1), (20, 5, 1), (10, 5, 0)])
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}volatility5{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], w) for dk in self.volatility5_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      volatility5_dict):
        for dk in self.volatility5_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                volatility5_df = kl_pd[day_ind - day_th + 1 - window:day_ind +
                                       1 - window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                volatility5_df = combine_kl_pd[
                    -day_th - window:] if combine_kl_pd.shape[0] > (
                        day_th + window) else combine_kl_pd
            volatility5 = calc_volatility5(volatility5_df,
                                           xd=dk[0],
                                           drift=dk[1],
                                           ewm=True if dk[2] == 1 else False)
            volatility5_score = volatility5.close
            volatility5_score = 0 if np.isnan(volatility5_score) else round(
                volatility5_score, g_point_num)
            volatility5_dict['{}volatility5{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], window)] = volatility5_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        volatility5_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               volatility5_dict=volatility5_dict)
        return volatility5_dict
