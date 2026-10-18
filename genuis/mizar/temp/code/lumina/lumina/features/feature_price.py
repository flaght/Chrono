# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from ultron.kdutils import regression


class FeaturePrice(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.price_rank_keys = frozenset([60, 90, 120, 252])
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_rank{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk, w)
            for dk in self.price_rank_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      price_rank_dict):
        for dk in self.price_rank_keys:
            day_th = dk
            if day_ind - day_th - window >= 0:
                price_rank_df = kl_pd[day_ind - day_th + 1 - window:day_ind +
                                      1 - window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                price_rank_df = combine_kl_pd[
                    -day_th - window:] if combine_kl_pd.shape[0] > (
                        day_th + window) else combine_kl_pd
            price_rank = price_rank_df.close.rank(
            )[-1] / price_rank_df.close.rank().shape[0]
            price_rank = 0 if np.isnan(price_rank) else round(
                price_rank, g_point_num)
            price_rank_dict['{}price_rank{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk,
                window)] = price_rank

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        price_rank_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               price_rank_dict=price_rank_dict)
        return price_rank_dict
