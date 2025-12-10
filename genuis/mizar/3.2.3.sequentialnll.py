import copy
import numpy as np
import torch
import random
from dotenv import load_dotenv

load_dotenv()
from lib.HybridTransformer.transformer import SequentialNLLTransformer,TemporientTransformer
from lib.uvx import *
from lib.syn005.trainer import Trainer as AETrainer
from lib.syn007.trainer import Trainer as STTrainer
from lib.syn007.evaluator import Evaluator
from lib.svx001 import scale_factors
from kdutils.macro2 import *
from kdutils.tactix import Tactix



'''
    
def load_autocoder_data(method, instruments, task_id, period,
                          nan_threshold, var_threshold, corr_threshold,
          ic_threshold, outdirs):
    FEATURE_PARAMS = {
        'nan_threshold':nan_threshold,
        'var_threshold':var_threshold,
        'corr_threshold':corr_threshold,
        'ic_threshold':ic_threshold
    }

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

    temp_outdirs = os.path.join(outdirs, "temp_data", "ae-st")
    if not os.path.exists(temp_outdirs):
        os.makedirs(temp_outdirs)
    filename = os.path.join(temp_outdirs, "{0}.feather".format(name))
    autocode_data = pd.read_feather(filename)
    pdb.set_trace()
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

    AUTOENCODE_PARAMS,TRAIN_PARAMS = load_params(file_dirs=outdirs, name="autoencode", model_name='params1', train_name="params1")
    #AUTOENCODE_PARAMS = {'d_model':48,'n_heads':4,'e_layers':2,
    #                     'd_ff':192,'dropout':0.25,'activation':'gelu',
    #                     'masking_ratio':0.25}
    #TRAIN_PARAMS = {'seq_len':45,'batch_size':256,'learning_rate':0.0003,
    #                'epochs':100,'patience':15,'device':'cuda:0'}
    
    AUTOENCODE_PARAMS['enc_in'] = feature_dim
    # pdb.set_trace()
    TOTAL_PARAMS = copy.deepcopy(AUTOENCODE_PARAMS)
    TOTAL_PARAMS.update(TRAIN_PARAMS)
    TOTAL_PARAMS.update(FEATURE_PARAMS)
    # pdb.set_trace()
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
'''

def set_random_seed(seed=42):
    """
    设置随机种子以确保训练可重复性
    解决不同训练run之间预测偏差反转的问题
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_autocoder_data(autocode_data):
    """数据校验函数"""
    logger.rule("Autoencoder 数据校验")
    
    # 1. 检查数据形状
    logger.print(f"数据形状: {autocode_data.shape}")
    
    # 2. 检查缺失值
    missing = autocode_data.isnull().sum()
    if missing.any():
        logger.print(f"⚠️ 发现缺失值:\n{missing[missing > 0]}")
    else:
        logger.print("✅ 无缺失值")
    
    # 3. 检查特征范围
    factor_cols = [c for c in autocode_data.columns if c.startswith('factor_')]
    if not factor_cols:
        logger.print("⚠️ 未找到 factor_ 开头的特征列")
        return False

    stats = autocode_data[factor_cols].describe().T
    logger.print(f"\n特征统计摘要 (前5个):")
    logger.print(stats.head())
    
    # 4. 检查是否坍缩 (方差极小)
    low_var_cols = stats[stats['std'] < 1e-6].index
    if len(low_var_cols) > 0:
        logger.print(f"\n⚠️ 警告: {len(low_var_cols)} 个特征方差极小 (可能坍缩):")
        logger.print(low_var_cols[:5].tolist())
    
    # 5. 检查是否所有值相同
    unique_counts = autocode_data[factor_cols].nunique()
    constant_cols = unique_counts[unique_counts <= 1].index
    if len(constant_cols) > 0:
        logger.print(f"\n❌ 严重错误: {len(constant_cols)} 个特征为常数 (完全坍缩):")
        logger.print(constant_cols[:5].tolist())
        return False

    return True

def fetch_autocoder_data(method, instruments, task_id, period,
                          nan_threshold, var_threshold, corr_threshold,
          ic_threshold, outdirs, force_update=False,
          data_source='train'):
    
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
    feature_dim = len(selected_features)

    AUTOENCODE_PARAMS,TRAIN_PARAMS = load_params(file_dirs=outdirs, name="autoencode", model_name='params1', train_name="params1")

    AUTOENCODE_PARAMS['enc_in'] = feature_dim
    TOTAL_PARAMS = copy.deepcopy(AUTOENCODE_PARAMS)
    TOTAL_PARAMS.update(TRAIN_PARAMS)
    TOTAL_PARAMS.update(FEATURE_PARAMS)

    name = create_train_records(method=method,task_id=task_id,instruments=instruments,period=period,
                         category='autoencode',params=TOTAL_PARAMS)
    
    temp_outdirs = os.path.join(outdirs, "temp_data", "ae-st")
    if not os.path.exists(temp_outdirs):
        os.makedirs(temp_outdirs)
        
    logger.rule("autoencode 构建特征")

    filename = os.path.join(temp_outdirs, "{0}_{1}.feather".format(name, data_source))
    if os.path.exists(filename) and not force_update:
        logger.print(f"Loading existing TRAIN autocoder data from {filename}")
        try:
            autocode_data = pd.read_feather(filename)
            return autocode_data
        except Exception as e:
            logger.print(f"Error loading file: {e}. Regenerating...")

    ### 预测
    trainer = AETrainer(params=AUTOENCODE_PARAMS, train_params=TRAIN_PARAMS,
                        output_dirs=outdirs, name=name)
    
    if data_source=='train':
        features_data, _ = fetch_clean_data2(method=method,task_id=task_id,instruments=instruments,
            output=outdirs, params={'nan_threshold':nan_threshold,'var_threshold':var_threshold})
    else:
        _, features_data = fetch_clean_data2(method=method,task_id=task_id,instruments=instruments,
            output=outdirs, params={'nan_threshold':nan_threshold,'var_threshold':var_threshold})
        
    logger.rule("Autoencoder Training & Feature Generation")
    trainer = AETrainer(params=AUTOENCODE_PARAMS, train_params=TRAIN_PARAMS,
                        output_dirs=outdirs, name=name)

    X, y, dates = trainer.prepare_data(features_data, selected_features, "nxt1_ret_{}h".format(period))

    fetures_samples = trainer.create_rolling_window_samples(X)
    ###认为模型必须存在，因为上游已经选好了存在的模型
    fetures_loader = trainer.create_predict_data_loader(fetures_samples)
    logger.panel("开始生成隐层特征...", title="特征生成")

    factors_array, _, _ = trainer.predict(
        model_method=TemporientTransformer,
        data_loader=fetures_loader,
        multi_timestep_extraction=False
    )

    logger.print(f"Generated Factors Shape: {factors_array.shape}")
    autocode_data = pd.DataFrame(factors_array, 
                                 columns=[f'factor_{i}' for i in range(factors_array.shape[1])])
    seq_len = TRAIN_PARAMS['seq_len']
    start_idx = seq_len - 1
    aligned_y = features_data["nxt1_ret_{}h".format(period)].values[start_idx:]
    aligned_dates = features_data['trade_time'].values[start_idx:]
    
    autocode_data['nxt1_ret_{0}h'.format(period)] = aligned_y
    autocode_data['trade_time'] = aligned_dates

    # 校验原始生成的数据
    if not validate_autocoder_data(autocode_data):
        logger.print("⚠️ 原始特征数据校验失败，可能存在模型坍缩！")

    
    logger.print("Applying Rolling Z-Score Standardization...")
    factor_cols = [c for c in autocode_data.columns if c.startswith('factor_')]
    for col in factor_cols:
        print(col)
        scale_factors(predict_data=autocode_data,
                      method='roll_zscore',
                      win=15,
                      factor_name=col)
        autocode_data[col] = autocode_data['transformed']
        autocode_data.drop(['transformed'], axis=1, inplace=True)

     # 处理标准化产生的 NaN (前 win-1 行)
    original_len = len(autocode_data)
    autocode_data = autocode_data.dropna()
    logger.print(f"移除 NaN 行: {original_len} → {len(autocode_data)} (移除了前 {original_len - len(autocode_data)} 行)")
    
    # 重置索引
    autocode_data.reset_index(drop=True, inplace=True)
    temp_outdirs = os.path.join(outdirs, "temp_data", "ae-st")
    if not os.path.exists(temp_outdirs):
        os.makedirs(temp_outdirs)
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
    
    autocode_data = fetch_autocoder_data(method=method,instruments=instruments,task_id=task_id,
                          period=period,nan_threshold=nan_threshold, 
                          var_threshold=var_threshold, corr_threshold=corr_threshold,
                          ic_threshold=ic_threshold, outdirs=outdirs, data_source='train',
                          force_update=False)
    pdb.set_trace()
    factor_features = [c for c in autocode_data.columns if c.startswith('factor_')]
    feature_dim = len(factor_features)

    logger.panel("Training SequentialNLLTransformer...", title="Step 4")

    MODEL_PARAMS,TRAIN_PARAMS = load_params(file_dirs=outdirs, name="sequentialnll", model_name='params1', train_name="params1")
    MODEL_PARAMS['enc_in'] = feature_dim
    MODEL_PARAMS['dec_in'] = feature_dim

    MODEL_PARAMS['output_variance'] = True

    TRAIN_PARAMS['loss_func'] = 'gaussian_nll'

    TOTAL_PARAMS = copy.deepcopy(MODEL_PARAMS)
    TOTAL_PARAMS.update(TRAIN_PARAMS)
    TOTAL_PARAMS.update(FEATURE_PARAMS)

    name = create_train_records(method=method,task_id=task_id,instruments=instruments,period=period,
                         category='sequentialnll',params=TOTAL_PARAMS)
    pdb.set_trace()
    # 设置随机种子以确保训练可重复性
    set_random_seed(42)

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

    trainer.train_model(model_method=SequentialNLLTransformer,train_loader=trainer_loader, val_loader=val_loader)

    logger.rule("训练集+校验集评估 (fitting_evaluate)")

    #train_loader = trainer.create_predict_data_loader(X_train_samples)
    train_loader = trainer.create_train_data_loader(X_train_samples, y_train_samples)
    pred_train, var_train, _ = trainer.predict(
        model_method=SequentialNLLTransformer,
        data_loader=train_loader
    )

    #val_loader = trainer.create_predict_data_loader(X_val_samples)
    val_loader = trainer.create_train_data_loader(X_val_samples,y_val_samples)
    pred_val, var_val, _ = trainer.predict(
        model_method=SequentialNLLTransformer,
        data_loader=val_loader
    )

    evaluator = Evaluator(
        resampling_win=period,
        roll_win=240,
        scale_method="roll_zscore"
    )
    
    evaluator.fitting_evaluate(
        y_train_true=y_train_samples,
        y_train_pred=pred_train,
        y_val_true=y_val_samples,
        y_val_pred=pred_val,
        var_train=var_train,
        var_val=var_val,
    )

    


def predict_model(method, task_id, instruments, period, name, nan_threshold, 
                var_threshold, corr_threshold, ic_threshold):
    
    FEATURE_PARAMS = {
        'nan_threshold':nan_threshold,
        'var_threshold':var_threshold,
        'corr_threshold':corr_threshold,
        'ic_threshold':ic_threshold
    }
    outdirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period), "research")
    
    autocode_data = fetch_autocoder_data(method=method,instruments=instruments,task_id=task_id,
                          period=period,nan_threshold=nan_threshold, 
                          var_threshold=var_threshold, corr_threshold=corr_threshold,
                          ic_threshold=ic_threshold, outdirs=outdirs, data_source='test',force_update=False)
    
    factor_features = [c for c in autocode_data.columns if c.startswith('factor_')]
    feature_dim = len(factor_features)

    logger.panel("Predicting with SequentialGaussianTransformer...", title="Step 5")

    MODEL_PARAMS,TRAIN_PARAMS = load_params(file_dirs=outdirs, name="sequentialnll", model_name='params1', train_name="params1")
    MODEL_PARAMS['enc_in'] = feature_dim
    MODEL_PARAMS['dec_in'] = feature_dim
    MODEL_PARAMS['output_variance'] = True
    TRAIN_PARAMS['loss_func'] = 'gaussian_nll'

    TOTAL_PARAMS = copy.deepcopy(MODEL_PARAMS)
    TOTAL_PARAMS.update(TRAIN_PARAMS)
    TOTAL_PARAMS.update(FEATURE_PARAMS)

    name = create_train_records(method=method,task_id=task_id,instruments=instruments,period=period,
                         category='sequentialnll',params=TOTAL_PARAMS)
    
    trainer = STTrainer(params=MODEL_PARAMS, train_params=TRAIN_PARAMS,output_dirs=outdirs,
              name=name)

    trainer.validate_data(autocode_data=autocode_data)

    X, y, dates = trainer.prepare_data(autocode_data, factor_features, "nxt1_ret_{}h".format(period))

    test_samples = trainer.create_rolling_window_samples(X)

    test_loader = trainer.create_predict_data_loader(test_samples)
    predictions, variances, _ = trainer.predict(model_method=SequentialNLLTransformer, data_loader=test_loader)

    aligned_dates = dates[TRAIN_PARAMS['seq_len']-1:]
    aligned_y = y[TRAIN_PARAMS['seq_len']-1:]

    result_df = pd.DataFrame({
        'trade_time': aligned_dates,
        'label': aligned_y,
        'prediction': predictions,
        'variance': variances
    })


    evaluator = Evaluator(
        resampling_win=period,
        roll_win=120,
        scale_method="roll_zscore"
    )

    returns_df = autocode_data[['trade_time', f'nxt1_ret_{period}h']].copy()
    returns_df = returns_df.set_index('trade_time')

    evaluator.final_evaluate(
        y_test_true=aligned_y,
        y_test_pred=predictions,
        var_test=variances,
        dates_test=aligned_dates,
        returns=returns_df,
        period=period,
    )

if __name__ == '__main__':
    variant = Tactix().start()
    train_model(method=variant.method, instruments=variant.instruments,
                    task_id=variant.task_id, period=variant.period,
                    name=variant.name, nan_threshold=0.5,
                   var_threshold=1e-10,corr_threshold=0.95,
                    ic_threshold=0.01)
    predict_model(method=variant.method, instruments=variant.instruments,
                    task_id=variant.task_id, period=variant.period,
                    name=variant.name, nan_threshold=0.5,
                    var_threshold=1e-10,corr_threshold=0.95,
                    ic_threshold=0.01)