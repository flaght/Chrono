# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_bias


class FeatureBias(FeatureBase, BuyFeatureMixin, SellFeatureMixin):
    """价格偏离率特征，支持买入，卖出"""

    def __init__(self):
        self.bias_keys = frozenset([(7, 1), (14, 1), (7, 0)])  # xd, ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_bias{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1], w)
            for dk in self.bias_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      bias_dict):
        for dk in self.bias_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                bias_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                bias_df = combine_kl_pd[-day_th -
                                        window:] if combine_kl_pd.shape[0] > (
                                            day_th + window) else combine_kl_pd
            bias = calc_bias(bias_df,
                             xd=dk[0],
                             ewm=True if dk[1] == 1 else False)
            bias_score = bias.close
            bias_score = 0 if np.isnan(bias_score) else round(
                bias_score, g_point_num)
            bias_dict['{}price_bias{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                window)] = bias_score
        return bias_dict

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        bias_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               bias_dict=bias_dict)
        return bias_dict
