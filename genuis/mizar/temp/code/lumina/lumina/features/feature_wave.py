# -*- encoding:utf-8 -*-
import numpy as np
from lumina.env import g_max_window, g_point_num
from ultron.ump.core.fixes import xrange
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical.wave import calc_wave


class FeatureWave(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.wave_keys = frozenset([21, 42, 60])
        self.windows = [i for i in range(g_max_window + 1)]
        self.wave_key_cnt = 3

    def get_feature_keys(self, buy_feature):
        return [
            '{}wave_score{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), xd_ind, w)
            for xd_ind in list(range(1, self.wave_key_cnt + 1))
            for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      wave_dict):
        for dk in self.wave_xd:
            day_th = dk * 2 + 1
            if day_ind - day_th - window >= 0:
                wave_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                                window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                wave_df = combine_kl_pd[-day_th -
                                        window:] if combine_kl_pd.shape[0] > (
                                            day_th + window) else combine_kl_pd

            for xd_ind in xrange(1, self.wave_key_cnt + 1):
                wave = calc_wave(wave_df, xd=xd_ind * dk)
                wave_score = wave.score
                wave_score = 0 if np.isnan(wave_score) else round(
                    wave_score, g_point_num)
                wave_dict['{}wave_score{}_w{}'.format(
                    self.feature_prefix(buy_feature=buy_feature), xd_ind,
                    window)] = wave_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        wave_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               wave_dict=wave_dict)
        return wave_dict
