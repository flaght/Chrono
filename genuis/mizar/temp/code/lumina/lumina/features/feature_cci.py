# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from ultron.ump.core.fixes import xrange
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_cci


class FeatureCCI(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.cci_keys = frozenset([(14, 1), (42, 1), (14, 0)])  # (xd, ewm)
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_cci{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1], w)
            for dk in self.cci_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      cci_dict):
        for dk in self.cci_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                cci_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                cci_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd
            cci = calc_cci(cci_df, xd=dk[0], ewm=True if dk[1] == 1 else False)
            cci_score = cci.close
            cci_score = 0 if np.isnan(cci_score) else round(
                cci_score, g_point_num)
            cci_dict['{}price_cci{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                window)] = cci_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        cci_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               cci_dict=cci_dict)
        return cci_dict
