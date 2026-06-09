import os,json,pdb,copy
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from kdutils.macro2 import *
from lib.composite.loader import DataLoader
from lib.composite.cleaner import DataCleaner
from lib.composite.feature import Featurer
from lib.syn001.linear import train_model1 as linear_train_model
from lib.syn001.linear import train_model2 as linear_equal_model
from lib.syn001.lasso import train_model1 as lasso_train_model
from lib.syn001.rigde import train_model1 as rigde_train_model
from lib.syn001.lassocv import train_model1 as lassocv_train_model
from lib.utils.params import Params
from lib.uvx import *


from kdutils.tactix import Tactix



def select_features(outdirs, feature_id):
    filename = os.path.join(outdirs, "selection", str(feature_id), "selected_features.feather")
    selected_features = pd.read_feather(filename)
    return selected_features.tolist()


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
        file_dirs=outdirs, name="linear", model_name='params1', 
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


def train_model(method, instruments, task_id, period, name, model_name):
    outdirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period), "research")
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
    #selected_features = features_df['feature'].tolist()[-1:]
    '''
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
    
    '''
    selected_features = ["DELTA(5,MT3(120,MCPS(120,MA(120,'pct_change'))))",
                        "MMASSI(90,'twap',MADecay(15,'order_flow_imbanlace_1'))",
                        "RSI(120,MCPS(120,RSI(120,'pct_change_close')))",
                        "MT3(90,MCPS(15,MSUM(120,'smart_tick_in')))"
                        ]
    

    #selected_features = ["MT3(90,MCPS(15,MSUM(120,'smart_tick_in')))"]
    
    
    #pdb.set_trace()
    train_data, test_data = fetch_clean_data2(method=method,task_id=task_id,instruments=instruments,
        output=outdirs, params=DATA_PARAMS)

    #pdb.set_trace()
    TOTAL_PARAMS = copy.deepcopy(MODEL_PARAMS)
    TOTAL_PARAMS.update(DATA_PARAMS)

    create_train_records(method=method,task_id=task_id,instruments=instruments,period=period,
                         category=model_name, params=TOTAL_PARAMS)
    
    if  model_name == 'linear':
        linear_train_model(train_data=train_data, test_data=test_data, 
                        selected_features=selected_features, 
                        params=MODEL_PARAMS,
                        roll_win=15, period=period, outdirs=os.path.join(outdirs,"result", model_name, str(int(Params.create_tag(TOTAL_PARAMS)))
                        )
                    )
    elif model_name == 'lasso':
        lasso_train_model(train_data=train_data, test_data=test_data,
                        selected_features=selected_features, 
                        params=MODEL_PARAMS,
                        roll_win=15, period=period,
                        outdirs=os.path.join(outdirs,"result", model_name, str(int(Params.create_tag(TOTAL_PARAMS)))
                        )
                    )
    elif model_name == 'rigde':
        rigde_train_model(train_data=train_data, test_data=test_data,
                        selected_features=selected_features, 
                        params=MODEL_PARAMS,
                        roll_win=15, period=period,
                        outdirs=os.path.join(outdirs,"result", model_name, str(int(Params.create_tag(TOTAL_PARAMS)))
                        )
                    )

    elif model_name == 'equal':
        linear_equal_model(train_data=train_data, test_data=test_data,
                        selected_features=selected_features, 
                        roll_win=15, period=period,
                        outdirs=os.path.join(outdirs,"result", model_name, str(int(Params.create_tag(TOTAL_PARAMS))))
                        )
        



                



    


if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == 'preprocess':
        preprocess_data(method=variant.method, instruments=variant.instruments,
                        task_id=variant.task_id, period=variant.period,
                        name=variant.name)
    elif variant.form == 'train':
        train_model(method=variant.method, instruments=variant.instruments,
                    task_id=variant.task_id, period=variant.period,
                    name=variant.name, model_name=variant.model_name)
