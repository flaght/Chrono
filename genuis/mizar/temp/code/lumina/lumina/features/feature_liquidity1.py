# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_liquidity1


class FeatureLiquidity1(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.liquidity1_keys = frozenset([(3, 1)])  # xd, ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}volume_liquidity1{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1], w)
            for dk in self.liquidity1_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      annealn_dict):
        for dk in self.liquidity1_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                annealn_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                   window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                annealn_df = combine_kl_pd[-day_th -
                                           window:] if combine_kl_pd.shape[
                                               0] > (day_th +
                                                     window) else combine_kl_pd
            annealn = calc_liquidity1(annealn_df,
                                      xd=dk[0],
                                      ewm=True if dk[1] == 1 else False)
            annealn_score = annealn.close
            annealn_score = 0 if np.isnan(annealn_score) else round(
                annealn_score, g_point_num)
            annealn_dict['{}volume_liquidity1{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                window)] = annealn_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        annealn_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               annealn_dict=annealn_dict)
        return annealn_dict
