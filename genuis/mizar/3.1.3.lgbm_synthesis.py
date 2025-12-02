import os,json,pdb,copy
import pandas as pd
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


def preprocess_data(method, instruments, task_id, period, name, 
          nan_threshold, var_threshold, corr_threshold,
          ic_threshold):

    logger.configure(log_file="./filename.log")
    outdirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period),"research")
    if not os.path.exists(outdirs):
        os.makedirs(outdirs)
    loader = DataLoader() ## 加载数据
    ### 加载所有数据，进行预处理
    train_data,val_data,test_data = loader.load_from_project(method=method, task_id=task_id, 
                                    instruments=instruments, 
                                    period=period, name=name)
    pdb.set_trace()
    ### 特征评估基于 训练集 + 校验集
    ### 清洗数据基于 训练集 + 校验集 + 测试集
    final_data = pd.concat([train_data, val_data, test_data],axis=0).sort_values(by=['trade_time','code'])
    factors_data = pd.concat([train_data, val_data],axis=0).sort_values(by=['trade_time','code'])

    cleaner = DataCleaner(
            nan_threshold=nan_threshold,
            var_threshold=var_threshold,
            target_col="nxt1_ret_{}h".format(period)
        )
    
    
    final_data = cleaner.clean(final_data)
    factors_data = cleaner.clean(factors_data)


    ### 保存用于后面训练
    save_clean_data(output=outdirs, data=final_data, 
                    params={'nan_threshold':nan_threshold,'var_threshold':var_threshold})

    ### 用于特征功能 不能接触测试集
    engineer = Featurer(corr_threshold=corr_threshold,
                        ic_threshold=ic_threshold,
                        target_col="nxt1_ret_{}h".format(period))
    
    selected_features, ic_dict = engineer.select_features(factors_data)
    feature_df = pd.DataFrame({'feature': selected_features})
    ic_df = pd.DataFrame({'feature': list(ic_dict.keys()),'IC': list(ic_dict.values())}).sort_values('IC', ascending=False)

    params = Params(base_path=os.path.join(outdirs), experiment_name="feature")
    
    params.save_params_with_content(params={'nan_threshold':nan_threshold,'var_threshold':var_threshold,
                                        'corr_threshold':corr_threshold,'ic_threshold':ic_threshold},
                        artifacts={'feature':feature_df,
                                   "ic":ic_df})


def train_model(method, instruments, task_id, period, name, 
          nan_threshold, var_threshold, corr_threshold,
          ic_threshold):
    

    FEATURE_PARAMS = {
        'nan_threshold':nan_threshold,
        'var_threshold':var_threshold,
        'corr_threshold':corr_threshold,
        'ic_threshold':ic_threshold
    }
    outdirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period), "research")
    
    factors_data = DataLoader().load_from_project(method=method, task_id=task_id, 
                                    instruments=instruments, 
                                    period=period, name=name)
    returns_data = factors_data[['trade_time','code', "nxt1_ret_{0}h".format(period)]].set_index(['trade_time','code'])["nxt1_ret_{0}h".format(period)]
    code = returns_data.index.get_level_values('code')[0]
    
    LGB_PARAMS,TRAIN_PARAMS = load_params(file_dirs=outdirs, name="lgbm", model_name='params1', train_name="params1")


    features_df = fetch_research_fetures(
        method=method, instruments=instruments,task_id=task_id,
        period=period, name='feature', 
        params=FEATURE_PARAMS)
    selected_features = features_df['feature'].tolist()
    
    
    train_data, test_data = fetch_clean_data2(method=method,task_id=task_id,instruments=instruments,
        output=outdirs, params={'nan_threshold':nan_threshold,'var_threshold':var_threshold})


    TOTAL_PARAMS = copy.deepcopy(LGB_PARAMS)
    TOTAL_PARAMS.update(TRAIN_PARAMS)
    TOTAL_PARAMS.update(FEATURE_PARAMS)

    create_train_records(method=method,task_id=task_id,instruments=instruments,period=period,
                         category='lgbm',params=TOTAL_PARAMS)

    trainer = Trainer(params=LGB_PARAMS, train_params=TRAIN_PARAMS)

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
    
    Evaluator(resampling_win=period, roll_win=240,scale_method='roll_zscore').fitting_metrics(
        train_factors=train_factors,val_factors=val_factors,returns=returns_data,
        period=period
    )

    logger.rule("以下是测试集信息")
    X_test, y_test, date_test = trainer.prepare_data(test_data, selected_features, "nxt1_ret_{}h".format(period))
    y_test_pred = trainer.predict(X_test, model)

    test_factors = pd.Series(y_test_pred, index=pd.MultiIndex.from_arrays(
        [date_test, [code] * len(date_test)],        # 传入两层索引的数据
        names=['trade_time', 'code']),    # 为每一层索引命名
          name='transformed')
    
    Evaluator().final_evaluate(
        y_test_true=y_test,
        y_test_pred=y_test_pred
    )

    Evaluator(resampling_win=period, roll_win=240,scale_method='roll_zscore').final_metrics(
        test_factors=test_factors,
        returns=returns_data,period=period
    )


    


if __name__ == '__main__':
    variant = Tactix().start()

    preprocess_data(method=variant.method, instruments=variant.instruments,
                    task_id=variant.task_id, period=variant.period,
                    name=variant.name, nan_threshold=0.5,
                    var_threshold=1e-10,corr_threshold=0.95,
                    ic_threshold=0.01)
