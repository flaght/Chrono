import pdb
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv()

from lib.HybridTransformer.transformer import TemporientTransformer
from lib.lsx001 import fetch_times
from kdutils.macro2 import *


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

    features_df = final_data

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")

    pdb.set_trace()
    masking_ratio = 0.25
    batch_size = 32
    seq_len = 240  # 时间序列长度
    feature_dim = len(final_data.columns)  # 每个时间点的特征数
    d_model = 128  # 模型的隐藏维度，也将是最终因子的维度
    learning_rate = 0.001
    epochs = 50  # 真实场景需要更多 epoch

    features_df = features_df.dropna()
    features_np = features_df.to_numpy(dtype=np.float32)

    FEATURE_DIM = features_np.shape[1]
    print(f"Data shape after cleaning: {features_np.shape}")

    # 3. 创建滚动窗口样本
    print(f"Creating rolling window samples with sequence length {seq_len}...")
    samples_np = create_rolling_window_samples(features_np, seq_len)
    print(f"Created {samples_np.shape[0]} samples.")
    pdb.set_trace()
    # 4. 创建 PyTorch DataLoader
    # 在自监督学习中，输入和目标是相同的
    samples_tensor = torch.from_numpy(samples_np)
    dataset = TensorDataset(samples_tensor, samples_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TemporientTransformer(enc_in=feature_dim,
                                  d_model=d_model,
                                  masking_ratio=masking_ratio).to(DEVICE)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (batch_inputs, batch_targets) in enumerate(dataloader):
            # 将数据移动到指定设备
            batch_inputs = batch_inputs.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)
            # 清零梯度
            optimizer.zero_grad()

            # 前向传播 (is_training=True 是模型内部的默认或需要手动设置)
            # 我们的 TemporiorientTransformer 实现中，训练时自动遮盖
            _enc_out, _dec_out, outputs = model(batch_inputs)

            loss = criterion(outputs, batch_targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                 print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Avg Loss: {total_loss / (i+1):.6f}")

        print(f"--- Epoch [{epoch+1}/{epochs}] Complete --- Avg Reconstruction Loss: {total_loss / len(dataloader):.6f} ---")
    print("✅ TemporiorientTransformer pre-training complete.")

    torch.save(model.state_dict(), 'temporiorient_encoder_pretrained.pth')
    print("Model saved to temporiorient_encoder_pretrained.pth")

method = 'bicso0'
instruments = 'rbb'
period = 5
name = 'lgbm'
task_id = str(INDEX_MAPPING[INSTRUMENTS_CODES[instruments]])

train_model(method=method,
            task_id=task_id,
            instruments=instruments,
            period=period)
