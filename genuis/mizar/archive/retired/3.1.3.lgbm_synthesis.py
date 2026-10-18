import os,json,pdb,copy,optuna
import pandas as pd
import numpy as np
from typing import Optional, Iterable,List
from dotenv import load_dotenv
load_dotenv()

from kdutils.macro2 import *
from lib.composite.loader import DataLoader
from lib.composite.cleaner import DataCleaner
from lib.composite.feature import Featurer
from lib.utils.params import Params
from lib.uvx import *
from lib.syn004.trainer import Trainer
from lib.syn004.evaluator import Evaluator

from kdutils.tactix import Tactix



def select_features(outdirs, feature_id):
    filename = os.path.join(outdirs, "selection", str(feature_id), "selected_features.feather")
    selected_features = pd.read_feather(filename)
    return selected_features.tolist()

def preprocess_data(method, instruments, task_id, period, name):

    outdirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period),"research")
    if not os.path.exists(outdirs):
        os.makedirs(outdirs)

    _,_,DATA_PARAMS = load_params1(
        file_dirs=outdirs, name="lgbm", model_name='params1', 
        train_name="params1", data_name="params1")

    if int(DATA_PARAMS['feature_id']) != 0:
        features_list = select_features(outdirs=outdirs, feature_id=DATA_PARAMS['feature_id'])
    else:
        features_list = []
    loader = DataLoader() ## 加载数据
    ### 加载所有数据，进行预处理, 加载指定筛选特征表
    pdb.set_trace()

    TOTAL_PARAMS = copy.deepcopy(DATA_PARAMS)
    pdb.set_trace()
    create_train_records(method=method,task_id=task_id,instruments=instruments,
                        period=period,category='lgbm',params=TOTAL_PARAMS)


    train_data,val_data,test_data = loader.load_from_project(
                                    method=method, task_id=task_id, 
                                    instruments=instruments, 
                                    period=period, name=name,
                                    features=features_list)
    ### 特征评估基于 训练集 + 校验集
    ### 清洗数据基于 训练集 + 校验集 + 测试集
    pdb.set_trace()
    final_data = pd.concat([train_data, val_data, test_data],axis=0).sort_values(by=['trade_time','code'])
    factors_data = pd.concat([train_data, val_data],axis=0).sort_values(by=['trade_time','code'])

    cleaner = DataCleaner(
            nan_threshold=float(DATA_PARAMS['nan_threshold']),
            var_threshold=float(DATA_PARAMS['var_threshold']),
            target_col="nxt1_ret_{}h".format(period)
        )
    
    
    final_data = cleaner.clean(final_data)
    factors_data = cleaner.clean(factors_data)


    ### 保存用于后面训练
    save_clean_data(output=outdirs, data=final_data, 
                    params=DATA_PARAMS)
    ### 用于特征功能 不能接触测试集
    engineer = Featurer(corr_threshold=float(DATA_PARAMS['corr_threshold']),
                        ic_threshold=float(DATA_PARAMS['ic_threshold']),
                        target_col="nxt1_ret_{0}h".format(period),
                        )
    
    ### 根据数据特性再做一次处理
    selected_features, ic_dict = engineer.select_features(
        df=factors_data,ic_threshold=float(DATA_PARAMS['ic_threshold']),
        roll_win=int(DATA_PARAMS['roll_win']),
        resampling_win=int(DATA_PARAMS['resampling_win']),
        method=DATA_PARAMS['ic_method'])

    feature_df = pd.DataFrame({'feature': selected_features})
    ic_df = pd.DataFrame({'feature': list(ic_dict.keys()),'IC': list(ic_dict.values())}).sort_values('IC', ascending=False)

    params = Params(base_path=os.path.join(outdirs), experiment_name="feature")
    
    params.save_params_with_content(params=DATA_PARAMS,
                        artifacts={'feature':feature_df,
                                   "ic":ic_df})

def train_model(method, instruments, task_id, period, name):
    outdirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period), "research")
    
    MODEL_PARAMS,TRAIN_PARAMS,DATA_PARAMS = load_params1(
        file_dirs=outdirs, name="lgbm", model_name='params1', 
        train_name="params1", data_name="params1")

    if int(DATA_PARAMS['feature_id']) != 0:
        features_list = select_features(outdirs=outdirs, feature_id=DATA_PARAMS['feature_id'])
    else:
        features_list = []
    
    train_data,val_data,_ = DataLoader().load_from_project(method=method, task_id=task_id, 
                                    instruments=instruments, 
                                    period=period, name=name,
                                    features=features_list)
    
    factors_data = pd.concat([train_data, val_data],axis=0).sort_values(by=['trade_time','code'])
    returns_data = factors_data[['trade_time','code', "nxt1_ret_{0}h".format(period)]].set_index(['trade_time','code'])["nxt1_ret_{0}h".format(period)]
    code = returns_data.index.get_level_values('code')[0]


    features_df = fetch_research_fetures(
        method=method, instruments=instruments,task_id=task_id,
        period=period, name='feature', 
        params=DATA_PARAMS)
    selected_features = features_df['feature'].tolist()

    selected_features = ["MMASSI(120,MNPOSITIVE(90,'corr_vwap_ask_price_0'),WMA(5,'smart_tick_in'))",
                        "MVHF(10,MMASSI(120,MPRO(60,MVHF(10,MPRO(60,'money'))),MAPOSITIVE(10,'twap')))",
                        "DELTA(120,MMAX(90,DELTA(90,'low')))",
                        "RSI(120,MCPS(120,RSI(120,'pct_change_close')))",
                        "MMAX(30,MQUANTILE(240,MMASSI(30,DELTA(90,MMIN(30,MIR(60,'smart_volume_out'))),MIChimoku(30,'mid_price_bias','delta_volume_bid1'))))",
                        "DELTA(90,MMaxDiff(60,MADecay(120,MADecay(120,MADecay(120,'smart_tick_in_pct')))))",
                        "MT3(90,MCPS(15,MSUM(120,'smart_tick_in')))",
                        "MT3(30,MT3(90,MARGMIN(120,'twap')))",
                        "MIR(10,MCORR(10,'pct_change_set','depth_imbalance_2'))",
                        "MMASSI(90,'twap',MADecay(15,'order_flow_imbanlace_1'))",
                        "MSharp(120,MRes(60,'smart_tick_out',MADiff(30,'delta_volume_ask1')),MCPS(120,'twap'))",
                        "MINIMUM(MMedian(15,SIGLOG2ABS('pct_change_set')),SIGLOG2ABS('corr_ret_ask_price_0'))",
                        "MCoef(5,MMaxDiff(5,'tick_out'),MKURT(10,'mid_price_bias_ratio'))",
                        "MRANK(120, SUBBED(MRANK(30, DELTA(60, 'high')), MRANK(20, DELTA(5, 'high'))))",
                        "DELTA(5,MT3(120,MCPS(120,MA(120,'pct_change'))))",
                        "MA(120,MADiff(15,EMA(60,MSUM(60,'smart_money_in_pct'))))",
                        "MIR(120,DELTA(90,MSKEW(90,DELTA(90,'high'))))",
                        "MIR(120,MPERCENT(90,DELTA(90,'twap')))",
                        "MHMA(30,MMedian(60,'smart_volume_in'))",
                        "MRANK(30,MQUANTILE(15,MOD('pct_change','order_flow_imbanlace_avg5')))",
                        "MMeanRes(60,MDEMA(30,'order_imbalance_ratio1'),MADecay(5,'order_imbalance_ratio1'))",
                        "RSI(120,MCPS(120,MMedian(90,'pct_change_set')))",
                        "MIR(15,DELTA(90,MA(90,DELTA(90,'twap'))))",
                        "MMedian(90,MVHF(10,MT3(120,'pct_change')))",
                        "DELTA(90,MMaxDiff(90,DELTA(90,SHIFT(60,'low'))))"]
    selected_features = [
      "RSI(120,MCPS(120,RSI(120,'pct_change_close')))",
      "MIR(120,DELTA(90,MSKEW(90,DELTA(90,'high'))))",
      "DELTA(5,MT3(120,MCPS(120,MA(120,'pct_change'))))",
      "MIR(15,DELTA(90,MA(90,DELTA(90,'twap'))))"
  ]
    selected_features = ["MIR(120,DELTA(90,MSKEW(90,DELTA(90,'high'))))", "MMedian(90,MVHF(10,MT3(120,'pct_change')))", "MVHF(10,MMASSI(120,MPRO(60,MVHF(10,MPRO(60,'money'))),MAPOSITIVE(10,'twap')))", "MINIMUM(MMedian(15,SIGLOG2ABS('pct_change_set')),SIGLOG2ABS('corr_ret_ask_price_0'))", "MT3(90,MCPS(15,MSUM(120,'smart_tick_in')))", "DELTA(120,MMAX(90,DELTA(90,'low')))", "MIR(15,DELTA(90,MA(90,DELTA(90,'twap'))))", "MT3(30,MT3(90,MARGMIN(120,'twap')))", "MMAX(30,MQUANTILE(240,MMASSI(30,DELTA(90,MMIN(30,MIR(60,'smart_volume_out'))),MIChimoku(30,'mid_price_bias','delta_volume_bid1'))))", "DELTA(90,MMaxDiff(60,MADecay(120,MADecay(120,MADecay(120,'smart_tick_in_pct')))))", "MRANK(30,MQUANTILE(15,MOD('pct_change','order_flow_imbanlace_avg5')))", "MIR(120,MPERCENT(90,DELTA(90,'twap')))", "DELTA(90,MMaxDiff(90,DELTA(90,SHIFT(60,'low'))))", "RSI(120,MCPS(120,MMedian(90,'pct_change_set')))", "MRANK(120, SUBBED(MRANK(30, DELTA(60, 'high')), MRANK(20, DELTA(5, 'high'))))", "MMASSI(90,'twap',MADecay(15,'order_flow_imbanlace_1'))", "MIR(10,MCORR(10,'pct_change_set','depth_imbalance_2'))", "MA(120,MADiff(15,EMA(60,MSUM(60,'smart_money_in_pct'))))", "MCoef(5,MMaxDiff(5,'tick_out'),MKURT(10,'mid_price_bias_ratio'))", "RSI(120,MCPS(120,RSI(120,'pct_change_close')))"]
    pdb.set_trace()
    train_data, test_data = fetch_clean_data2(method=method,task_id=task_id,instruments=instruments,
        output=outdirs, params=DATA_PARAMS)


    TOTAL_PARAMS = copy.deepcopy(MODEL_PARAMS)
    TOTAL_PARAMS.update(TRAIN_PARAMS)
    TOTAL_PARAMS.update(DATA_PARAMS)

    create_train_records(method=method,task_id=task_id,instruments=instruments,period=period,
                         category='lgbm',params=TOTAL_PARAMS)

    trainer = Trainer(params=MODEL_PARAMS, train_params=TRAIN_PARAMS)

    X, y, dates = trainer.prepare_data(train_data, selected_features, "nxt1_ret_{}h".format(period))

    X_train, X_val, y_train, y_val, dates_train, dates_val = trainer.split_data(
        X, y, dates, train_ratio=0.7)
    
    model = trainer.train_single(X_train, y_train,
                                 X_val=X_val, y_val=y_val,
                                 selected_features=selected_features)
    
    y_train_pred = trainer.predict(X_train, model)
    y_val_pred = trainer.predict(X_val, model)
    ## 模型评估
    Evaluator().fitting_evaluate(
        y_train_true=y_train,
        y_train_pred=y_train_pred,
        y_val_true=y_val,
        y_val_pred=y_val_pred
    )
    ## ER 评估
    val_factors = pd.Series(y_val_pred, index=pd.MultiIndex.from_arrays(
        [dates_val, [code] * len(dates_val)],        # 传入两层索引的数据
        names=['trade_time', 'code']    # 为每一层索引命名
        ), name='transformed')
    
    train_factors = pd.Series(y_train_pred, index=pd.MultiIndex.from_arrays(
        [dates_train, [code] * len(dates_train)],        # 传入两层索引的数据
        names=['trade_time', 'code']),    # 为每一层索引命名
          name='transformed')
    
    Evaluator(resampling_win=period, roll_win=0, scale_method='raw').fitting_metrics(
        train_factors=train_factors,val_factors=val_factors,returns=returns_data,
        period=period
    )

    logger.rule("以下是测试集信息")
   
    X_test, y_test, date_test = trainer.prepare_data(test_data, selected_features, "nxt1_ret_{}h".format(period))
    y_test_pred = trainer.predict(X_test, model)
    test_returns = test_data[['trade_time','code', "nxt1_ret_{0}h".format(period)]].set_index(['trade_time','code'])["nxt1_ret_{0}h".format(period)]

    test_factors = pd.Series(y_test_pred, index=pd.MultiIndex.from_arrays(
        [date_test, [code] * len(date_test)],        # 传入两层索引的数据
        names=['trade_time', 'code']),    # 为每一层索引命名
          name='transformed')
    
    Evaluator().final_evaluate(
        y_test_true=y_test,
        y_test_pred=y_test_pred
    )

    Evaluator(resampling_win=period, roll_win=period, scale_method='roll_zscore').final_metrics(
        test_factors=test_factors,
        returns=test_returns,period=period
    )


def _objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                    test_data: pd.DataFrame, period:int,code:str,
                    X_val: Optional[np.ndarray] = None,
                    y_val: Optional[np.ndarray] = None,
                    selected_features: Optional[List[str]] = None):
    FIXED_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'seed': 42,
        'n_jobs': -1}
    
    TRAIN_PARAMS = {
        'num_boost_round': 500,        # 对应 yaml 配置
        'early_stopping_rounds': 300,  # 对应 yaml 配置
        'verbose_eval': False,         # 寻优时关闭日志刷屏
        'scale': 10000                 # 对应 yaml 配置
    }

    param_grid = {
        # 核心正则化参数
        'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 15.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 32),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 200, 1000),
        # 采样参数
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        
        # 其他
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        'path_smooth': trial.suggest_float('path_smooth', 0.0, 5.0), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True) 
    }
    model_params = {**FIXED_PARAMS, **param_grid}
    trainer = Trainer(params=model_params, train_params=TRAIN_PARAMS)
    model = trainer.train_single(
        X_train=X_train, 
        y_train=y_train,
        X_val=X_val, 
        y_val=y_val,
        selected_features=selected_features
    )

    X_test, y_test, date_test = trainer.prepare_data(
        test_data, selected_features, "nxt1_ret_{}h".format(period))
    
    y_test_pred = trainer.predict(X_test, model)
    test_returns = test_data[['trade_time','code', "nxt1_ret_{0}h".format(period)]].set_index(['trade_time','code'])["nxt1_ret_{0}h".format(period)]
    test_factors = pd.Series(y_test_pred, index=pd.MultiIndex.from_arrays(
        [date_test, [code] * len(date_test)],        # 传入两层索引的数据
        names=['trade_time', 'code']),    # 为每一层索引命名
          name='transformed')
    stats_test = Evaluator(resampling_win=period, roll_win=period, scale_method='roll_zscore').final_metrics(
        test_factors=test_factors,
        returns=test_returns,period=period
    )
    is_profit_positive = (stats_test['avg_ret'] > 0) and (stats_test['sharpe2'] > 0)
    is_ic_negative = (stats_test['ic_mean'] <= 0.0001) 
    if is_profit_positive and is_ic_negative:
        # 触发惩罚逻辑
        trial.set_user_attr("penalty", "overfitting_suspected")
        trial.set_user_attr("real_ic", stats_test['ic_mean'])
        trial.set_user_attr("real_sharpe", stats_test['sharpe2'])
        
        # 返回极低值，确保这组参数永远不会被选为最优
        # 注意：这里返回4个 -999，对应 4 个 maximize 目标
        return (-999.0, -999.0, -999.0, -999.0)
    trial.set_user_attr("penalty", "none")
    return stats_test['calmar'], stats_test['avg_ret'], stats_test['sharpe2'],stats_test['ic_mean']


def optuna_model(method, instruments, task_id, period, name):
    pdb.set_trace()
    outdirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period), "research")
    
    MODEL_PARAMS,TRAIN_PARAMS,DATA_PARAMS = load_params1(
        file_dirs=outdirs, name="lgbm", model_name='params1', 
        train_name="params1", data_name="params1")

    if int(DATA_PARAMS['feature_id']) != 0:
        features_list = select_features(outdirs=outdirs, feature_id=DATA_PARAMS['feature_id'])
    else:
        features_list = []
    
    train_data,val_data,_ = DataLoader().load_from_project(method=method, task_id=task_id, 
                                    instruments=instruments, 
                                    period=period, name=name,
                                    features=features_list)
    
    factors_data = pd.concat([train_data, val_data],axis=0).sort_values(by=['trade_time','code'])
    returns_data = factors_data[['trade_time','code', "nxt1_ret_{0}h".format(period)]].set_index(['trade_time','code'])["nxt1_ret_{0}h".format(period)]
    code = returns_data.index.get_level_values('code')[0]


    features_df = fetch_research_fetures(
        method=method, instruments=instruments,task_id=task_id,
        period=period, name='feature', 
        params=DATA_PARAMS)
    selected_features = features_df['feature'].tolist()
    selected_features = ["MIR(120,DELTA(90,MSKEW(90,DELTA(90,'high'))))", "MMedian(90,MVHF(10,MT3(120,'pct_change')))", "MVHF(10,MMASSI(120,MPRO(60,MVHF(10,MPRO(60,'money'))),MAPOSITIVE(10,'twap')))", "MINIMUM(MMedian(15,SIGLOG2ABS('pct_change_set')),SIGLOG2ABS('corr_ret_ask_price_0'))", "MT3(90,MCPS(15,MSUM(120,'smart_tick_in')))", "DELTA(120,MMAX(90,DELTA(90,'low')))", "MIR(15,DELTA(90,MA(90,DELTA(90,'twap'))))", "MT3(30,MT3(90,MARGMIN(120,'twap')))", "MMAX(30,MQUANTILE(240,MMASSI(30,DELTA(90,MMIN(30,MIR(60,'smart_volume_out'))),MIChimoku(30,'mid_price_bias','delta_volume_bid1'))))", "DELTA(90,MMaxDiff(60,MADecay(120,MADecay(120,MADecay(120,'smart_tick_in_pct')))))", "MRANK(30,MQUANTILE(15,MOD('pct_change','order_flow_imbanlace_avg5')))", "MIR(120,MPERCENT(90,DELTA(90,'twap')))", "DELTA(90,MMaxDiff(90,DELTA(90,SHIFT(60,'low'))))", "RSI(120,MCPS(120,MMedian(90,'pct_change_set')))", "MRANK(120, SUBBED(MRANK(30, DELTA(60, 'high')), MRANK(20, DELTA(5, 'high'))))", "MMASSI(90,'twap',MADecay(15,'order_flow_imbanlace_1'))", "MIR(10,MCORR(10,'pct_change_set','depth_imbalance_2'))", "MA(120,MADiff(15,EMA(60,MSUM(60,'smart_money_in_pct'))))", "MCoef(5,MMaxDiff(5,'tick_out'),MKURT(10,'mid_price_bias_ratio'))", "RSI(120,MCPS(120,RSI(120,'pct_change_close')))"]
    pdb.set_trace()
    train_data, test_data = fetch_clean_data2(method=method,task_id=task_id,instruments=instruments,
        output=outdirs, params=DATA_PARAMS)


    TOTAL_PARAMS = copy.deepcopy(MODEL_PARAMS)
    TOTAL_PARAMS.update(TRAIN_PARAMS)
    TOTAL_PARAMS.update(DATA_PARAMS)

    ### 准备数据源
    temp_trainer = Trainer(params={}, train_params={})
    X, y, dates = temp_trainer.prepare_data(train_data, selected_features, "nxt1_ret_{}h".format(period))

    X_train, X_val, y_train, y_val, dates_train, dates_val = temp_trainer.split_data(
        X, y, dates, train_ratio=0.7
    )
    study_name = f"lgbm_multi_{task_id}"
    study = optuna.create_study(
        study_name=study_name,
        load_if_exists=True,
        directions=["maximize", "maximize", "maximize", "maximize"]
    )
    N_TRIALS = 50
    study.optimize(lambda trial: _objective(trial=trial,X_train=X_train,y_train=y_train,
                test_data=test_data,period=period,code=code,
                X_val=X_val,y_val=y_val,
                selected_features=selected_features),
                n_trials=N_TRIALS,
            show_progress_bar=True)

    


if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == 'preprocess':
        preprocess_data(method=variant.method, instruments=variant.instruments,
                        task_id=variant.task_id, period=variant.period,
                        name=variant.name)
    elif variant.form == 'train':
        train_model(method=variant.method, instruments=variant.instruments,
                    task_id=variant.task_id, period=variant.period,
                    name=variant.name)
    elif variant.form == 'optuna':
        optuna_model(method=variant.method, instruments=variant.instruments,
                    task_id=variant.task_id, period=variant.period,
                    name=variant.name)
