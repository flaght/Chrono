# -*- encoding:utf-8 -*-
import pdb
import numpy as np
from lumina.env import g_max_window, g_point_num
from lumina.features.fixes import BuyFeatureMixin, SellFeatureMixin, FeatureBase
from lumina.techinical import calc_efi


class FeatureEFI(FeatureBase, BuyFeatureMixin, SellFeatureMixin):

    def __init__(self):
        self.efi_keys = frozenset([(10, 1, 1), (13, 1, 1),
                                   (13, 1, 0)])  # xd drift ewn
        self.windows = [i for i in range(g_max_window + 1)]

    def get_feature_keys(self, buy_feature):
        return [
            '{}volume_efi{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], w) for dk in self.efi_keys for w in self.windows
        ]

    def _calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature, window,
                      efi_dict):
        for dk in self.efi_keys:
            day_th = dk[0] * 2 + 1
            if day_ind - day_th - window >= 0:
                efi_df = kl_pd[day_ind - day_th + 1 - window:day_ind + 1 -
                               window]
            else:
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                efi_df = combine_kl_pd[-day_th -
                                       window:] if combine_kl_pd.shape[0] > (
                                           day_th + window) else combine_kl_pd
            efi = calc_efi(efi_df,
                           xd=dk[0],
                           drift=dk[1],
                           ewm=True if dk[2] == 1 else False)
            efi_score = efi.close
            efi_score = 0 if np.isnan(efi_score) else round(
                efi_score, g_point_num)
            efi_dict['{}volume_efi{}_{}_{}_w{}'.format(
                self.feature_prefix(buy_feature=buy_feature), dk[0], dk[1],
                dk[2], window)] = efi_score

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        efi_dict = {}
        for w in self.windows:
            self._calc_feature(kl_pd=kl_pd,
                               combine_kl_pd=combine_kl_pd,
                               day_ind=day_ind,
                               buy_feature=buy_feature,
                               window=w,
                               efi_dict=efi_dict)
        return efi_dict
