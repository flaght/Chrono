import pdb, optuna
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from kdutils.composite.models.base import Base


class NativeBayesClassifier(Base):

    def __init__(self, instruments, names, strategy_settings, key):
        super(NativeBayesClassifier, self).__init__(name='random_forest')
        self.instruments = instruments
        self.names = names
        self.strategy_settings = strategy_settings
        self.key = key
        self.task_name = "{0}_{1}".format(self.name, key) if isinstance(
            key, str) else self.name
        self.neutral_threshold = self.strategy_settings['commission'] * 0.05
        self.target_map = {-1: 0, 0: 1, 1: 2}
        self.reversed_map = {v: k for k, v in self.target_map.items()}

    def standard_features(self, features):
        # CategoricalNB 要求输入特征是 "非负整数"。
        return features[self.names].replace(self.target_map)

    def stranard_target(self, target):
        return target

    def normalize_target(self, raw_meta_signal):
        raw_meta_signal = raw_meta_signal
        return raw_meta_signal

    def _fit(self, X_train_fold, y_train_fold, params, **kwargs):
        X_train_fold = self.standard_features(X_train_fold)
        model = CategoricalNB(**params)
        model.fit(X_train_fold, y_train_fold)
        return model

    def _predict(self, model, X_fold):
        X_fold = self.standard_features(X_fold)
        predicted_labels = model.predict(X_fold)
        return predicted_labels

    ### 参数搜索空间
    def objective_cv(self, trial: optuna.Trial, X_train_val: pd.DataFrame,
                     y_train_val: pd.Series,
                     total_data_train_val: pd.DataFrame,
                     strategy_settings: dict) -> float:

        params = {'alpha': trial.suggest_float('alpha', 1e-5, 10.0, log=True)}

        return self.objective_cv1(X_train_val=X_train_val,
                                  y_train_val=y_train_val,
                                  total_data_train_val=total_data_train_val,
                                  strategy_settings=strategy_settings,
                                  params=params)

    ### 参数寻优评估
    def optuna(self, train_data, val_data, train_positions, val_positions):
        return super(NativeBayesClassifier,
                     self).optuna(train_data=train_data,
                                  val_data=val_data,
                                  train_positions=train_positions,
                                  val_positions=val_positions,
                                  objective_func=self.objective_cv)

    ## 参数确定后训练模型
    def train(self, train_data, val_data, train_positions, val_positions,
              params):
        X_train_val, y_train_val = self.prepare1(
            train_data=train_data,
            val_data=val_data,
            train_positions=train_positions,
            val_positions=val_positions,
            neutral_threshold=self.neutral_threshold)

        ## 训练模型
        model = self._fit(X_train_fold=X_train_val,
                          y_train_fold=y_train_val,
                          params=params)
        return model

    ### 预测
    def predict(self, model, train_positions, val_positions, test_positions,
                name):
        positions = pd.concat([train_positions, val_positions, test_positions],
                              axis=0)
        positions = positions.set_index('trade_time')
        features = self.standard_features(positions[self.names])
        raw_meta_signal = model.predict(features)
        raw_meta_signal = pd.Series(raw_meta_signal, index=positions.index)
        raw_meta_signal.name = "{0}_{1}".format(self.name, name)
        return raw_meta_signal
