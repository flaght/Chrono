# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_ichimoku


class FeatureIchimoku(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.ichimoku_keys = frozenset([(9, 26, 52, 1), (9, 26, 52, 0)
                                        ])  # tenkan, kijun, senkou, ewm
        self.ichimoku_name = frozenset(['span_a', 'span_b'])
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}ichimoku{}_{}_{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], n, w) for dk in self.ichimoku_keys
            for n in self.ichimoku_name for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      ichimoku_dict):
        for dk in self.ichimoku_keys:
            day_th = dk[1] * 2 + 1
            if day_ind - day_th - window >= 0:
                ichimoku_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                    window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                ichimoku_df = combine_kl_pd[
                    -day_th - window:] if combine_kl_pd.shape[0] > (
                        day_th + window) else combine_kl_pd
            span_a, span_b = calc_ichimoku(ichimoku_df,
                                           tenkan=dk[0],
                                           kijun=dk[1],
                                           senkou=dk[2],
                                           ewm=True if dk[3] == 1 else False)

            span_a_score = span_a.close
            span_b_score = span_b.close

            span_a_score = 0 if np.isnan(span_a_score) else round(
                span_a_score, g_point_num)
            span_b_score = 0 if np.isnan(span_b_score) else round(
                span_b_score, g_point_num)

            ichimoku_dict['{}ichimoku{}_{}_{}_{}_span_a_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], window)] = span_a_score

            ichimoku_dict['{}ichimoku{}_{}_{}_{}_span_b_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], dk[3], window)] = span_b_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        ichimoku_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               ichimoku_dict=ichimoku_dict)
        return ichimoku_dict
