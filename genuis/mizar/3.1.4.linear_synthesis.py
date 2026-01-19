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


from kdutils.tactix import Tactix



def select_features(outdirs, feature_id):
    filename = os.path.join(outdirs, "selection", str(feature_id), "selected_features.feather")
    selected_features = pd.read_feather(filename)
    return selected_features.tolist()



def train_model(method, instruments, task_id, period, name):
    outdirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period), "research")
    model_name = 'linear'
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
    selected_features = features_df['feature'].tolist()
    
    pdb.set_trace()
    train_data, test_data = fetch_clean_data2(method=method,task_id=task_id,instruments=instruments,
        output=outdirs, params=DATA_PARAMS)


    TOTAL_PARAMS = copy.deepcopy(MODEL_PARAMS)
    TOTAL_PARAMS.update(DATA_PARAMS)

    create_train_records(method=method,task_id=task_id,instruments=instruments,period=period,
                         category=model_name,params=TOTAL_PARAMS)
    ### 
    pdb.set_trace()
    print('-->')
                



    


if __name__ == '__main__':
    variant = Tactix().start()
    if variant.form == 'train':
        train_model(method=variant.method, instruments=variant.instruments,
                    task_id=variant.task_id, period=variant.period,
                    name=variant.name)
