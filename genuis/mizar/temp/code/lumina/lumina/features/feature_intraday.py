# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_intraday


class FeatureIntraday(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.intraday_keys = frozenset([(7, 2, 1), (14, 2, 1),
                                        (7, 2, 0)])  #xd,drift,ewm
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}price_intraday{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], w) for dk in self.intraday_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      intraday_dict):
        for dk in self.intraday_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                intraday_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                    window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                intraday_df = combine_kl_pd[
                    -day_th - window:] if combine_kl_pd.shape[0] > (
                        day_th + window) else combine_kl_pd
            intraday = calc_intraday(intraday_df,
                                     xd=dk[0],
                                     drift=dk[1],
                                     ewm=True if dk[2] == 1 else False)
            intraday_score = intraday.close
            intraday_score = 0 if np.isnan(intraday_score) else round(
                intraday_score, g_point_num)
            intraday_dict['{}price_intraday{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], window)] = intraday_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        intraday_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               intraday_dict=intraday_dict)
        return intraday_dict
