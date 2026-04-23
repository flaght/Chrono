### 标准化指定因子 和绩效
import os,json,pdb,copy
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from kdutils.macro2 import *
from lib.utils.params import Params
from lib.composite.loader import DataLoader
from lib.composite.cleaner import DataCleaner
from lib.composite.feature import Featurer
from lib.syn001.linear import train_model1 as linear_train_model
from lib.iux002 import generate_simple_id
from lib.utils.params import Params
from lib.uvx import *
from lib.cux001 import FactorEvaluate1
from lumina.genetic.util import create_id

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
        file_dirs=outdirs, name="check", model_name='params1', 
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
                        period=period,category='linear',params=TOTAL_PARAMS)


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

def train_check(method, instruments, task_id, period, name):
    outdirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period), "research")
    model_name = 'check'
    MODEL_PARAMS,DATA_PARAMS = load_params2(
        file_dirs=outdirs, name=model_name, model_name='params1',
        data_name="params1")

    if int(DATA_PARAMS['feature_id']) != 0:
        features_list = select_features(outdirs=outdirs, feature_id=DATA_PARAMS['feature_id'])
    else:
        features_list = []
    
    train_data, val_data,_ = DataLoader().load_from_project(method=method, task_id=task_id, 
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
    pdb.set_trace()
    selected_features = features_df['feature'].tolist()
    
    pdb.set_trace()
    train_data, test_data = fetch_clean_data2(method=method,task_id=task_id,instruments=instruments,
        output=outdirs, params=DATA_PARAMS)

    name = Params.create_tag(DATA_PARAMS)

    outdirs_factors = os.path.join(outdirs, "check", str(name))
    if not os.path.exists(outdirs_factors):
        os.makedirs(outdirs_factors)
    for features in selected_features:
        factor_id = create_id(generate_simple_id(features))
        print(factor_id, features)
        dt1 = train_data[['trade_time','code', features, "nxt1_ret_{0}h".format(period)]]
        evaluate1 = FactorEvaluate1(factor_data=dt1,
                                factor_name=features,
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=15,
                                fee=0.000,
                                scale_method='raw',
                                expression=features,
                                resampling_win=period)

        dt2 = test_data[['trade_time','code', features, "nxt1_ret_{0}h".format(period)]]
        evaluate2 = FactorEvaluate1(factor_data=dt2,
                                factor_name=features,
                                ret_name='nxt1_ret_{0}h'.format(period),
                                roll_win=15,
                                fee=0.000,
                                scale_method='raw',
                                expression=features,
                                resampling_win=period)
        stats1 = evaluate1.run()
        stats2 = evaluate2.run()

        evaluate1.plot_results()
        evaluate1.save_results(os.path.join(outdirs_factors, str(factor_id), "train"))

        evaluate2.plot_results()
        evaluate2.save_results(os.path.join(outdirs_factors, str(factor_id), "test"))

        ## 图片统一保存
        evaluate1.figure.savefig(os.path.join(outdirs_factors, "plot", "train_{0}.png".format(factor_id)), dpi=300)
        evaluate2.figure.savefig(os.path.join(outdirs_factors, "plot", "test_{0}.png".format(factor_id)), dpi=300)



if __name__ == '__main__':
    variant = Tactix().start()
    pdb.set_trace()
    if variant.form == 'preprocess':
        preprocess_data(method=variant.method, instruments=variant.instruments,
                        task_id=variant.task_id, period=variant.period,
                        name=variant.name)
    elif variant.form == 'check':
        train_check(method=variant.method, instruments=variant.instruments,
                    task_id=variant.task_id, period=variant.period,
                    name=variant.name)