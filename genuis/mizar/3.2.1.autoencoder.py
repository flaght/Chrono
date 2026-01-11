import copy
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from lib.HybridTransformer.transformer import TemporientTransformer
from lib.uvx import *
from lib.syn005.trainer import Trainer
from lib.syn005.evaluator import Evaluator
from lib.uvx import *
from kdutils.macro2 import *
from kdutils.tactix import Tactix


def train_model(method, task_id, instruments, period, name,
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


    
    features_df = fetch_research_fetures(
        method=method, instruments=instruments,task_id=task_id,
        period=period, name='feature', 
        params=FEATURE_PARAMS)
    selected_features = features_df['feature'].tolist()
    train_data, _ = fetch_clean_data2(method=method,task_id=task_id,instruments=instruments,
        output=outdirs, params={'nan_threshold':nan_threshold,'var_threshold':var_threshold})
    
    feature_dim = len(selected_features) 
    # 指定GPU设备
    '''
    if torch.cuda.is_available():
        DEVICE = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
    else:
        DEVICE = 'cpu'

    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(gpu_id)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
    '''
    pdb.set_trace()
    AUTOENCODE_PARAMS,TRAIN_PARAMS = load_params(file_dirs=outdirs, name="autoencode", model_name='params1', train_name="params1")
    AUTOENCODE_PARAMS['enc_in'] = feature_dim

    TOTAL_PARAMS = copy.deepcopy(AUTOENCODE_PARAMS)
    TOTAL_PARAMS.update(TRAIN_PARAMS)
    TOTAL_PARAMS.update(FEATURE_PARAMS)

    name = create_train_records(method=method,task_id=task_id,instruments=instruments,period=period,
                         category='autoencode',params=TOTAL_PARAMS)

    # 理论分析
    compression_ratio_old = 128 / (240 * feature_dim)
    compression_ratio_new = AUTOENCODE_PARAMS['d_model'] / (TRAIN_PARAMS['seq_len'] * feature_dim)
    snr_gain = np.sqrt(240 / TRAIN_PARAMS['seq_len'])
    logger.panel(f"  压缩率: {compression_ratio_old:.2%} → {compression_ratio_new:.2%} "
                 f"(提升 {compression_ratio_new/compression_ratio_old:.1f}倍)"
                 f"  SNR增益: {snr_gain:.2f}倍 (序列长度减少的平方根)",
                 title="理论改进:")


    ### 切割训练集/校验集
    logger.rule("训练集+校验集 训练过程")
    
    trainer = Trainer(params=AUTOENCODE_PARAMS, train_params=TRAIN_PARAMS,output_dirs=outdirs, name=str(name))

    X, y, dates = trainer.prepare_data(train_data, selected_features, "nxt1_ret_{}h".format(period))

    X_train, X_val, y_train, y_val, dates_train, dates_val = trainer.split_data(
        X, y, dates, train_ratio=0.7)
    

    train_samples = trainer.create_rolling_window_samples(X_train)
    val_samples = trainer.create_rolling_window_samples(X_val)

    trainer_loader = trainer.create_train_data_loader(x_samples=train_samples, y_samples=train_samples)
    val_loader = trainer.create_train_data_loader(x_samples=val_samples, y_samples=val_samples)

    

    trainer.train_model(model_method=TemporientTransformer,train_loader=trainer_loader, val_loader=val_loader)

def predict_model(method, task_id, instruments, period, name,
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
    
    features_df = fetch_research_fetures(
        method=method, instruments=instruments,task_id=task_id,
        period=period, name='feature', 
        params=FEATURE_PARAMS)
    selected_features = features_df['feature'].tolist()
    _, test_data = fetch_clean_data2(method=method,task_id=task_id,instruments=instruments,
        output=outdirs, params={'nan_threshold':nan_threshold,'var_threshold':var_threshold})
    feature_dim = len(selected_features) 


    AUTOENCODE_PARAMS,TRAIN_PARAMS = load_params(file_dirs=outdirs, name="autoencode", model_name='params1', train_name="params1")
    AUTOENCODE_PARAMS['enc_in'] = feature_dim

    TOTAL_PARAMS = copy.deepcopy(AUTOENCODE_PARAMS)
    TOTAL_PARAMS.update(TRAIN_PARAMS)
    TOTAL_PARAMS.update(FEATURE_PARAMS)

    name = create_train_records(method=method,task_id=task_id,instruments=instruments,period=period,
                         category='autoencode',params=TOTAL_PARAMS)
    
    logger.rule("测试集 评估过程")
    
    trainer = Trainer(params=AUTOENCODE_PARAMS, train_params=TRAIN_PARAMS,output_dirs=outdirs, name=name)

    # 准备测试数据
    X, y, dates = trainer.prepare_data(test_data, selected_features, "nxt1_ret_{}h".format(period))
    
    # 创建滚动窗口样本
    test_samples = trainer.create_rolling_window_samples(X)
    
    test_loader = trainer.create_predict_data_loader(test_samples)
    
    # 生成隐层特征
    logger.panel("开始生成隐层特征...", title="特征生成")
    factors_array, original_array, reconstructed_array = trainer.predict(
        model_method=TemporientTransformer,
        data_loader=test_loader,
        multi_timestep_extraction=False  # 多时间步提取: * 3
    )
    

    # 对齐时间戳（滚动窗口会减少样本数）
    factor_timestamps = dates[TRAIN_PARAMS['seq_len'] - 1:]
    aligned_y = y[TRAIN_PARAMS['seq_len'] - 1:]
    
    logger.panel("开始评估 Autoencoder 模型质量...", title="模型评估")
    latent_features = factors_array  # 隐层特征
    target = aligned_y  # 目标收益率
    
    times = pd.to_datetime(factor_timestamps).time  # 时间
    evaluator = Evaluator()
    evaluator.final_metrics(
        latent_features=latent_features,
        target=target,
        times=times,
        original=original_array,  # 原始输入
        reconstructed=reconstructed_array,  # 重建输出
        standardize_windows=[15,60,240]
    )
    


if __name__ == '__main__':
    variant = Tactix().start()

    #train_model(method=variant.method, instruments=variant.instruments,
    #                task_id=variant.task_id, period=variant.period,
    #                name=variant.name, nan_threshold=0.5,
    #                var_threshold=1e-10,corr_threshold=0.95,
    #                ic_threshold=0.01)
    predict_model(method=variant.method, instruments=variant.instruments,
                    task_id=variant.task_id, period=variant.period,
                    name=variant.name, nan_threshold=0.5,
                    var_threshold=1e-10,corr_threshold=0.95,
                    ic_threshold=0.01)
