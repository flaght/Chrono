import pdb, optuna
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from kdutils.composite.models.base import Base


class LightBGMClassifier(Base):

    def __init__(self, instruments, names, strategy_settings, key):
        super(LightBGMClassifier, self).__init__(name='LBGM')
        self.instruments = instruments
        self.names = names
        self.strategy_settings = strategy_settings
        self.scaler = StandardScaler()
        self.key = key
        self.task_name = "{0}_{1}".format(self.name, key) if isinstance(
            key, str) else self.name
        self.neutral_threshold = self.strategy_settings['commission'] * 0.05
        self.target_map = {-1: 0, 0: 1, 1: 2}
        self.reversed_map = {v: k for k, v in self.target_map.items()}

    def standard_features(self, features):
        return features

    def stranard_target(self, target):
        target = target.map(self.target_map)
        return target

    def normalize_target(self, raw_meta_signal):
        raw_meta_signal = raw_meta_signal.map(self.reversed_map)
        return raw_meta_signal

    def _fit(self, X_train_fold, y_train_fold, params, **kwargs):
        pdb.set_trace()
        X_val_fold = kwargs['X_val_fold']
        y_val_fold = kwargs['y_val_fold']
        y_train_fold = self.stranard_target(y_train_fold)
        y_val_fold = self.stranard_target(y_val_fold)

        lgb_train = lgb.Dataset(X_train_fold, label=y_train_fold)
        lgb_val = lgb.Dataset(X_val_fold, label=y_val_fold)
        model = lgb.train(params,
                          lgb_train,
                          num_boost_round=1000,
                          valid_sets=[lgb_val],
                          valid_names=['val'],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
        return model

    def _predict(self, model, X_fold):
        predicted_probs = model.predict(X_fold,
                                        num_iteration=model.best_iteration)
        predicted_labels = np.argmax(predicted_probs, axis=1)
        return predicted_labels

    ### 参数搜索空间
    def objective_cv(self, trial: optuna.Trial, X_train_val: pd.DataFrame,
                     y_train_val: pd.Series,
                     total_data_train_val: pd.DataFrame,
                     strategy_settings: dict) -> float:
        # 1. 定义超参数搜索空间 (包含DART和强力正则化)
        params = {
            'objective':
            'multiclass',
            'num_class':
            3,
            'metric':
            'multi_logloss',
            'verbose':
            -1,
            'n_jobs':
            -1,
            #'device':'gpu',
            'seed':
            42,
            #'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            'learning_rate':
            trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves':
            trial.suggest_int('num_leaves', 7, 40),
            'lambda_l1':
            trial.suggest_float('lambda_l1', 1e-5, 10.0, log=True),
            'lambda_l2':
            trial.suggest_float('lambda_l2', 1e-5, 10.0, log=True),
            'feature_fraction':
            trial.suggest_float('feature_fraction', 0.5, 0.9),
            'min_child_samples':
            trial.suggest_int('min_child_samples', 50, 250),
        }
        return self.objective_cv1(X_train_val=X_train_val,
                                  y_train_val=y_train_val,
                                  total_data_train_val=total_data_train_val,
                                  strategy_settings=strategy_settings,
                                  params=params)

    ### 参数寻优评估
    def optuna(self, train_data, val_data, train_positions, val_positions):
        return super(LightBGMClassifier,
                     self).optuna(train_data=train_data,
                                  val_data=val_data,
                                  train_positions=train_positions,
                                  val_positions=val_positions,
                                  objective_func=self.objective_cv)

    ## 参数确定后训练模型
    def train(self, train_data, val_data, train_positions, val_positions,
              params):
        pdb.set_trace()
        X_train, y_train, X_val, y_val = self.prepare2(
            train_data=train_data,
            val_data=val_data,
            train_positions=train_positions,
            val_positions=val_positions,
            neutral_threshold=self.neutral_threshold)

        ## 训练模型
        model = self._fit(X_train_fold=X_train,
                          y_train_fold=y_train,
                          X_val_fold=X_val,
                          y_val_fold=y_val,
                          params=params)
        return model

    ### 预测
    def predict(self, model, train_positions, val_positions, test_positions,
                name):
        pdb.set_trace()
        positions = pd.concat([train_positions, val_positions, test_positions],
                              axis=0)
        positions = positions.set_index('trade_time')
        features = self.standard_features(positions[self.names])
        raw_meta_signal = model.predict(features)
        raw_meta_signal = pd.Series(raw_meta_signal, index=positions.index)
        raw_meta_signal.name = "random_forest_{0}".format(name)
        return raw_meta_signal
