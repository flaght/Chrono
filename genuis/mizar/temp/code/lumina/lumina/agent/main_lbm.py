# -*- encoding:utf-8 -*-
"""示例agent主裁LBM度模块"""
import pdb
from lumina.agent.fixes import MainBase, BuyAgentMixin
from lumina.features import *
from ultron.ump.agent.base import agent_main_make_xy
from ultron.ump.model.principles import Principles, princi_features

#g_feature_list = [FeaturePrice, FeatureDeg, FeatureStochRSI]
g_feature_list = [FeatureBias, FeaturePGO, FeatureWillr]


class LBMFiter(Principles):

    @agent_main_make_xy
    def make_xy(self, **kwarg):
        features_str = princi_features(g_feature_list, MainLBM, True)
        regex = 'result|{}'.format(features_str)
        lbm_df = self.order_has_ret.filter(regex=regex)
        return lbm_df


class MainLBM(MainBase, BuyAgentMixin):

    def _init_estimator(self, **kwarg):
        self.fiter().estimator.lgbm_classifier(
            boosting_type='gbdt',
            objective='binary',
            random_state=2021,
            #device='gpu',
            gpu_platform_id=1,
            gpu_device_id=1,
            learning_rate=0.5,
            n_estimators=200,
            reg_alpha=10,
            reg_lambda=60)

    def get_predict_col(self):
        ### 特征值返回
        features = princi_features(g_feature_list, MainLBM)
        return features

    def get_fiter_class(self):
        return LBMFiter

    @classmethod
    def class_unique_id(cls):
        return 'extend_main_lbm'
