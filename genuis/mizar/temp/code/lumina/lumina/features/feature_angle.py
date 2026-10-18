# -*- encoding:utf-8 -*-
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_angle


class FeatureAngle(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.angle_keys = frozenset([21, 42, 60])
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}deg_angle_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk, w)
            for dk in self.angle_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      deg_dict):
        for dk in self.angle_keys:
            if day_ind - dk - window >= 0:
                angle_pd = kl_pd[day_ind - dk + 1 - window:day_ind + 1 -
                                  window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                angle_pd = combine_kl_pd[
                    -dk - window:] if combine_kl_pd.shape[0] > (
                        dk + window) else combine_kl_pd
            ang = calc_angle(angle_pd, xd=dk)
            ang = 0 if np.isnan(ang) else round(ang, g_point_num)
            deg_dict['{}deg_ang{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk,
                window)] = ang

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        deg_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               deg_dict=deg_dict)
        return deg_dict
