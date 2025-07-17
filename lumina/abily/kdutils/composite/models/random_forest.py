import pdb, optuna
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifierImpl
from kdutils.composite.models.base import Base


class RandomForestClassifier(Base):

    def __init__(self, instruments, names, strategy_settings, key):
        super(RandomForestClassifier, self).__init__(name='random_forest')
        self.instruments = instruments
        self.names = names
        self.strategy_settings = strategy_settings
        self.scaler = StandardScaler()
        self.key = key
        self.task_name = "{0}_{1}".format(self.name, key) if isinstance(
            key, str) else self.name
        self.neutral_threshold = self.strategy_settings['commission'] * 0.05

    def standard_features(self, features):
        return self.scaler.fit_transform(features)

    def stranard_target(self, target):
        return target

    def normalize_target(self, raw_meta_signal):
        raw_meta_signal = raw_meta_signal
        return raw_meta_signal

    def _fit(self, X_train_fold, y_train_fold, params, **kwargs):
        X_train_fold = self.standard_features(X_train_fold)
        model = RandomForestClassifierImpl(**params)
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
        # --- 1. 定义超参数搜索空间 ---
        params = {
            # 树的数量
            'n_estimators':
            trial.suggest_int('n_estimators', 100, 500, step=50),

            # 控制单棵树的复杂度，防止过拟合
            'max_depth':
            trial.suggest_int('max_depth', 5, 20),
            'min_samples_leaf':
            trial.suggest_int('min_samples_leaf', 20, 200),
            'min_samples_split':
            trial.suggest_int('min_samples_split', 40, 400),
            # 控制随机性
            'max_features':
            trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.8]),

            # 其他参数
            'n_jobs':
            -1,
            'random_state':
            42,
            'class_weight':
            'balanced'
        }
        return self.objective_cv1(X_train_val=X_train_val,
                                  y_train_val=y_train_val,
                                  total_data_train_val=total_data_train_val,
                                  strategy_settings=strategy_settings,
                                  params=params)

    ### 参数寻优评估
    def optuna(self, train_data, val_data, train_positions, val_positions):
        return super(RandomForestClassifier,
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
