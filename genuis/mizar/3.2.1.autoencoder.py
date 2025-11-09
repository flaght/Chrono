import pdb, joblib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()

from lib.HybridTransformer.transformer import TemporientTransformer
from lib.lsx001 import fetch_times
from lib.svx001 import scale_factors
from kdutils.macro2 import *


def standard_features(prepare_features, method, win):
    features = prepare_features.columns
    predict_data = prepare_features.copy().dropna()
    for f in features:
        scale_factors(predict_data=predict_data,
                      method=method,
                      win=win,
                      factor_name=f)
        prepare_features[f] = predict_data['transformed']
    return prepare_features


def create_rolling_window_samples(data, seq_len):
    """
    将连续的时间序列数据转换为重叠的滚动窗口样本。
    
    Args:
        data (np.array): 输入的特征数据，形状为 (num_timesteps, num_features)。
        seq_len (int): 每个样本的时间窗口长度。
        
    Returns:
        np.array: 样本数据，形状为 (num_samples, seq_len, num_features)。
    """
    num_timesteps, num_features = data.shape
    num_samples = num_timesteps - seq_len + 1

    # 使用 numpy 的 stride_tricks 高效创建滚动窗口，避免循环
    shape = (num_samples, seq_len, num_features)
    strides = (data.strides[0], data.strides[0], data.strides[1])
    samples = np.lib.stride_tricks.as_strided(data,
                                              shape=shape,
                                              strides=strides)

    return samples


def train_model(method, task_id, instruments, period):
    time_array = fetch_times(method=method,
                             task_id=task_id,
                             instruments=instruments)
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))

    filename = os.path.join(dirs, "final_data.feather")
    final_data = pd.read_feather(filename).set_index(['trade_time', 'code'])
    print(final_data.columns)

    new_columns = [
        "f{0}".format(i) for i in range(0,
                                        len(final_data.columns) - 1)
    ]
    new_columns = new_columns + ["nxt1_ret_{}h".format(period)]
    final_data.columns = new_columns
    final_data = final_data.drop(["nxt1_ret_{}h".format(period)], axis=1)
    old_data = final_data.copy()
    ## 时序标准化
    standard_features(prepare_features=final_data,
                      method='roll_zscore',
                      win=15)

    features_df = final_data.dropna()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")

    masking_ratio = 0.25
    batch_size = 32
    seq_len = 240  # 时间序列长度
    feature_dim = len(final_data.columns)  # 每个时间点的特征数
    d_model = 128  # 模型的隐藏维度，也将是最终因子的维度
    learning_rate = 0.001
    epochs = 50  # 真实场景需要更多 epoch

    PATIENCE = 5  # 如果验证损失连续5个epoch没有改善，就停止训练
    best_val_loss = float('inf')
    patience_counter = 0

    ### 切割训练集/校验集
    pdb.set_trace()

    features_df = features_df.dropna()
    features_np = features_df.to_numpy(dtype=np.float32)

    train_data = features_df.loc[
        time_array['train_time'][0]:time_array['train_time'][1]]
    val_data = features_df.loc[
        time_array['val_time'][0]:time_array['val_time'][1]]

    train_np = train_data.to_numpy(dtype=np.float32)
    val_np = val_data.to_numpy(dtype=np.float32)

    FEATURE_DIM = features_np.shape[1]
    print(f"Data shape after cleaning: {features_np.shape}")

    # 3. 创建滚动窗口样本
    print(f"Creating rolling window samples with sequence length {seq_len}...")
    train_samples = create_rolling_window_samples(train_np, seq_len)
    val_samples = create_rolling_window_samples(val_np, seq_len)
    print(f"Created {train_samples.shape[0]} samples.")

    # 4. 创建 PyTorch DataLoader
    # 在自监督学习中，输入和目标是相同的
    train_dataset = TensorDataset(torch.from_numpy(train_samples),
                                  torch.from_numpy(train_samples))
    val_dataset = TensorDataset(torch.from_numpy(val_samples),
                                torch.from_numpy(val_samples))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False)  # 验证集不需要打乱

    model = TemporientTransformer(enc_in=feature_dim,
                                  d_model=d_model,
                                  masking_ratio=masking_ratio).to(DEVICE)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dirs = os.path.join("temp/auto")
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (batch_inputs, _) in enumerate(train_loader):
            # 将数据移动到指定设备
            batch_inputs = batch_inputs.to(DEVICE)
            # 清零梯度
            optimizer.zero_grad()

            # 前向传播 (is_training=True 是模型内部的默认或需要手动设置)
            # 我们的 TemporiorientTransformer 实现中，训练时自动遮盖
            _enc_out, _dec_out, outputs = model(batch_inputs)

            loss = criterion(outputs, batch_inputs)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Avg Loss: {total_loss / (i+1):.6f}"
                )
        # -- 验证阶段 --
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (batch_inputs, _) in enumerate(val_loader):
                batch_inputs = batch_inputs.to(DEVICE)
                _enc_out, _dec_out, outputs = model(
                    batch_inputs, masking_ratio=0)  # 重建任务，验证时也需要重建
                loss = criterion(outputs, batch_inputs)
                val_loss += loss.item()
                if (i + 1) % 100 == 0:
                    print(
                        f" Step [{i+1}/{len(val_loader)}], Avg Loss: {val_loss / (i+1):.6f}"
                    )

        avg_val_loss = val_loss / len(val_loader)

        # -- 早停逻辑 --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_temporiorient_model.pth')
            print(
                f"Validation loss improved. Saved best model to 'best_temporiorient_model.pth'"
            )
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

        filename = os.path.join(
            dirs, "temporiorient_encoder_pretrained_{0}.pth".format(epoch + 1))
        torch.save(model.state_dict(), filename)
        print(
            f"--- Epoch [{epoch+1}/{epochs}] Complete --- Avg Reconstruction Loss: {total_loss / len(train_loader):.6f} ---"
        )
    print("✅ TemporiorientTransformer pre-training complete.")

    torch.save(model.state_dict(), 'temporiorient_encoder_pretrained.pth')
    print("Model saved to temporiorient_encoder_pretrained.pth")


def predict_model(method, task_id, instruments, period):
    pdb.set_trace()
    model_path = os.path.join("temp", "auto",
                              "temporiorient_encoder_pretrained_18.pth")
    time_array = fetch_times(method=method,
                             task_id=task_id,
                             instruments=instruments)
    dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                        str(task_id), str(period))

    filename = os.path.join(dirs, "final_data.feather")
    final_data = pd.read_feather(filename).set_index(['trade_time', 'code'])
    print(final_data.columns)

    returns_data = final_data[["nxt1_ret_{}h".format(period)]]
    new_columns = [
        "f{0}".format(i) for i in range(0,
                                        len(final_data.columns) - 1)
    ]
    new_columns = new_columns + ["nxt1_ret_{}h".format(period)]
    final_data.columns = new_columns
    final_data = final_data.drop(["nxt1_ret_{}h".format(period)], axis=1)
    old_data = final_data.copy()

    feature_dim = len(final_data.columns)  # 每个时间点的特征数
    d_model = 128  # 模型的隐藏维度，也将是最终因子的维度
    seq_len = 240  # 时间序列长度
    batch_size = 32

    ## 时序标准化
    # 1. 加载标准化器并处理数据
    standard_features(prepare_features=final_data,
                      method='roll_zscore',
                      win=15)

    features_df = final_data.dropna()
    features_np = features_df.to_numpy(dtype=np.float32)

    test_data = features_df.loc[
        time_array['test_time'][0]:time_array['test_time'][1]]

    test_np = test_data.to_numpy(dtype=np.float32)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # 2. 加载模型
    print(f"Loading model from {model_path}")
    model = TemporientTransformer(enc_in=feature_dim,
                                  d_model=d_model).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()  # **非常重要：设置为评估模式**

    # 3. 创建滚动窗口样本
    print(f"Creating rolling window samples with sequence length {seq_len}...")
    samples_np = create_rolling_window_samples(test_np, seq_len)

    # 4. 创建 DataLoader
    dataset = TensorDataset(torch.from_numpy(samples_np))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_factors = []

    # 5. 模型推理
    print("Running model inference to generate factors...")
    with torch.no_grad():  # **非常重要：关闭梯度计算**
        for (batch_inputs, ) in dataloader:
            batch_inputs = batch_inputs.to(DEVICE)
            enc_out, _, _ = model(batch_inputs, masking_ratio=0.0)
            # 提取最后一个时间点的因子
            final_factors = enc_out[:, -1, :]  # Shape: [batch_size, d_model]

            all_factors.append(final_factors.cpu().numpy())

    # 6. 整合并格式化结果
    print("Formatting results...")
    factors_array = np.concatenate(all_factors, axis=0)
    pdb.set_trace()
    # 创建因子 DataFrame
    factor_columns = [f'factor_{i}' for i in range(d_model)]

    # 因子对应的时间戳是每个窗口的最后一个时间戳
    # data_df 的索引是从 0 开始的，长度为 N
    # samples_np 的长度是 N - seq_len + 1
    # 第一个 sample 对应 data_df 的第 seq_len-1 个时间点
    factor_timestamps = test_data.index[seq_len - 1:]

    factors_df = pd.DataFrame(
        factors_array,
        index=factor_timestamps,
        columns=factor_columns,
    )
    autoencod_data = pd.concat([factors_df, returns_data],axis=1).dropna()
    original_data = pd.concat([final_data.loc[time_array['test_time'][0]:time_array['test_time'][1]],returns_data],axis=1).dropna().reset_index()
    autoencod_data.reset_index().sort_values(by=['trade_time','code']).to_feather("autoencode_data.feather")
    original_data.sort_values(by=['trade_time','code']).to_feather("original_data.feather")
    pdb.set_trace()
    print("✅ Factor generation complete.")


method = 'cicso0'
instruments = 'ims'
period = 5
#name = 'lgbm'
task_id = str(INDEX_MAPPING[INSTRUMENTS_CODES[instruments]])

predict_model(method=method,
              task_id=task_id,
              instruments=instruments,
              period=period)
