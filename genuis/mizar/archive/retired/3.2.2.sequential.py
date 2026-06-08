import copy
import numpy as np
from dotenv import load_dotenv

load_dotenv()
from lib.HybridTransformer.transformer import SequentialTransformer, TemporientTransformer
from lib.uvx import *
from lib.syn005.trainer import Trainer as AETrainer
from lib.syn006.trainer import Trainer as STTrainer
from kdutils.macro2 import *
from kdutils.tactix import Tactix

def load_autocoder_data(method, instruments, task_id, period,
                          nan_threshold, var_threshold, corr_threshold,
          ic_threshold, outdirs):
    FEATURE_PARAMS = {
        'nan_threshold':nan_threshold,
        'var_threshold':var_threshold,
        'corr_threshold':corr_threshold,
        'ic_threshold':ic_threshold
    }
    #AUTOENCODE_PARAMS,TRAIN_PARAMS = load_params(file_dirs=outdirs, name="autoencode", model_name='params1', train_name="params1")
    AUTOENCODE_PARAMS = {'d_model':48,'n_heads':4,'e_layers':2,
                         'd_ff':192,'dropout':0.25,'activation':'gelu',
                         'masking_ratio':0.25}
    TRAIN_PARAMS = {'seq_len':45,'batch_size':256,'learning_rate':0.0003,
                    'epochs':100,'patience':15,'device':'cuda:0'}
    
    AUTOENCODE_PARAMS['enc_in'] = 130
    TOTAL_PARAMS = copy.deepcopy(AUTOENCODE_PARAMS)
    TOTAL_PARAMS.update(TRAIN_PARAMS)
    TOTAL_PARAMS.update(FEATURE_PARAMS)

    name = create_train_records(method=method,task_id=task_id,instruments=instruments,period=period,
                         category='autoencode',params=TOTAL_PARAMS)
    pdb.set_trace()
    temp_outdirs = os.path.join(outdirs, "temp_data", "ae-st")
    if not os.path.exists(temp_outdirs):
        os.makedirs(temp_outdirs)
    filename = os.path.join(temp_outdirs, "{0}.feather".format(name))
    autocode_data = pd.read_feather(filename)
    return autocode_data
    

def create_autocoder_data(method, instruments, task_id, period,
                          nan_threshold, var_threshold, corr_threshold,
          ic_threshold, outdirs):
    FEATURE_PARAMS = {
        'nan_threshold':nan_threshold,
        'var_threshold':var_threshold,
        'corr_threshold':corr_threshold,
        'ic_threshold':ic_threshold
    }
    
    features_df = fetch_research_fetures(
        method=method, instruments=instruments,task_id=task_id,
        period=period, name='feature', 
        params=FEATURE_PARAMS)
    selected_features = features_df['feature'].tolist()

    train_data, _ = fetch_clean_data2(method=method,task_id=task_id,instruments=instruments,
        output=outdirs, params={'nan_threshold':nan_threshold,'var_threshold':var_threshold})
    feature_dim = len(selected_features) 

    #AUTOENCODE_PARAMS,TRAIN_PARAMS = load_params(file_dirs=outdirs, name="autoencode", model_name='params1', train_name="params1")
    AUTOENCODE_PARAMS = {'d_model':48,'n_heads':4,'e_layers':2,
                         'd_ff':192,'dropout':0.25,'activation':'gelu',
                         'masking_ratio':0.25}
    TRAIN_PARAMS = {'seq_len':45,'batch_size':256,'learning_rate':0.0003,
                    'epochs':100,'patience':15,'device':'cuda:0'}
    
    AUTOENCODE_PARAMS['enc_in'] = feature_dim
    pdb.set_trace()
    TOTAL_PARAMS = copy.deepcopy(AUTOENCODE_PARAMS)
    TOTAL_PARAMS.update(TRAIN_PARAMS)
    TOTAL_PARAMS.update(FEATURE_PARAMS)
    pdb.set_trace()
    name = create_train_records(method=method,task_id=task_id,instruments=instruments,period=period,
                         category='autoencode',params=TOTAL_PARAMS)
    
    logger.rule("autoencode 构建特征")
    #name = '1087378733380904'
    trainer = AETrainer(params=AUTOENCODE_PARAMS, train_params=TRAIN_PARAMS,
                        output_dirs=outdirs, name=name)
    
    X, y, dates = trainer.prepare_data(train_data, selected_features, "nxt1_ret_{}h".format(period))

    # 创建滚动窗口样本
    test_samples = trainer.create_rolling_window_samples(X)
    
    test_loader = trainer.create_predict_data_loader(test_samples)
    logger.panel("开始生成隐层特征...", title="特征生成")

    factors_array, _, _ = trainer.predict(
        model_method=TemporientTransformer,
        data_loader=test_loader,
        multi_timestep_extraction=False
    )
    logger.print(f"Generated Factors Shape: {factors_array.shape}")

    autocode_data = pd.DataFrame(factors_array, 
                                 columns=[f'factor_{i}' for i in range(factors_array.shape[1])])
    seq_len = TRAIN_PARAMS['seq_len']
    start_idx = seq_len - 1
    aligned_y = train_data["nxt1_ret_{}h".format(period)].values[start_idx:]
    aligned_dates = train_data['trade_time'].values[start_idx:]
    
    autocode_data['nxt1_ret_{0}h'.format(period)] = aligned_y
    autocode_data['trade_time'] = aligned_dates

    temp_outdirs = os.path.join(outdirs, "temp_data", "ae-st")
    if not os.path.exists(temp_outdirs):
        os.makedirs(temp_outdirs)
    filename = os.path.join(temp_outdirs, "{0}.feather".format(name))
    autocode_data.to_feather(filename)
    return autocode_data

    



def train_model(method, task_id, instruments, period, name, nan_threshold, 
                var_threshold, corr_threshold, ic_threshold):
    
    FEATURE_PARAMS = {
        'nan_threshold':nan_threshold,
        'var_threshold':var_threshold,
        'corr_threshold':corr_threshold,
        'ic_threshold':ic_threshold
    }

    outdirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period), "research")
    
    autocode_data = load_autocoder_data(method=method,instruments=instruments,task_id=task_id,
                          period=period,nan_threshold=nan_threshold, 
                          var_threshold=var_threshold, corr_threshold=corr_threshold,
                          ic_threshold=ic_threshold, outdirs=outdirs)
    ### 
    pdb.set_trace()
    factor_features = [c for c in autocode_data.columns if c.startswith('factor_')]
    feature_dim = len(factor_features)

    logger.panel("Training SequentialTransformer...", title="Step 4")

    MODEL_PARAMS,TRAIN_PARAMS = load_params(file_dirs=outdirs, name="sequential", model_name='params1', train_name="params1")
    MODEL_PARAMS['enc_in'] = feature_dim
    MODEL_PARAMS['dec_in'] = feature_dim

    TOTAL_PARAMS = copy.deepcopy(MODEL_PARAMS)
    TOTAL_PARAMS.update(TRAIN_PARAMS)
    TOTAL_PARAMS.update(FEATURE_PARAMS)

    name = create_train_records(method=method,task_id=task_id,instruments=instruments,period=period,
                         category='sequential',params=TOTAL_PARAMS)

    
    trainer = STTrainer(params=MODEL_PARAMS, train_params=TRAIN_PARAMS,output_dirs=outdirs,
              name=name)

    X, y, dates = trainer.prepare_data(autocode_data, factor_features, "nxt1_ret_{}h".format(period))
    X_train, X_val, y_train, y_val, dates_train, dates_val = trainer.split_data(
        X, y, dates, train_ratio=0.7)

    X_train_samples = trainer.create_rolling_window_samples(X_train)
    X_val_samples = trainer.create_rolling_window_samples(X_val)

    y_train_samples = y_train[TRAIN_PARAMS['seq_len']-1:]
    y_val_samples = y_val[TRAIN_PARAMS['seq_len']-1:]

    trainer_loader = trainer.create_train_data_loader(x_samples=X_train_samples, y_samples=y_train_samples)
    val_loader = trainer.create_train_data_loader(x_samples=X_val_samples, y_samples=y_val_samples)

    trainer.train_model(model_method=SequentialTransformer,train_loader=trainer_loader, val_loader=val_loader)
    
    
    



if __name__ == '__main__':
    variant = Tactix().start()
    train_model(method=variant.method, instruments=variant.instruments,
                    task_id=variant.task_id, period=variant.period,
                    name=variant.name, nan_threshold=0.5,
                    var_threshold=1e-10,corr_threshold=0.95,
                    ic_threshold=0.01)