import re,os,copy,pdb
from typing import Tuple, List, Dict, Optional
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from lib import logger

def gaussian_nll_loss(pred_mean, pred_var, target,
                      lambda_diversity=1.0, lambda_pred_mean=0.0):
    """
    改进版：使用相对标准差计算多样性惩罚 + 添加预测均值约束

    Args:
        lambda_diversity: 方差多样性惩罚系数
        lambda_pred_mean: 预测均值约束系数 (方案C: 约束预测均值接近0)
    """
    # 转换参数类型
    lambda_diversity = float(lambda_diversity)
    lambda_pred_mean = float(lambda_pred_mean)

    if pred_mean.shape != target.shape:
        target = target.view_as(pred_mean)
    if pred_var.shape != target.shape:
        pred_var = pred_var.view_as(pred_mean)

    # NLL Loss (不需要裁剪)
    nll = 0.5 * (torch.log(pred_var) + (target - pred_mean).pow(2) / pred_var)
    nll_loss = nll.mean()

    # 方差多样性鼓励 (使用相对标准差)
    var_mean = torch.mean(pred_var)
    var_std = torch.std(pred_var)

    # 相对标准差: std / mean，范围在 0~1 之间
    # 加 1e-8 防止除零
    relative_std = var_std / (var_mean + 1e-8)

    # 多样性惩罚: 鼓励高相对标准差
    diversity_penalty = -lambda_diversity * relative_std

    # 方案C: 预测均值约束 - 惩罚预测均值偏离0
    # 这比直接约束 bias 更合理，因为它约束的是整体预测分布
    pred_mean_penalty = lambda_pred_mean * pred_mean.mean().pow(2)

    return nll_loss + diversity_penalty + pred_mean_penalty



class Trainer(object):
    def __init__(self, params: Dict = None, train_params: Dict = None, output_dirs:str = None, name=None):
        self.model = None
        self.train_params = train_params
        self.params = params
        self.name = name
        self.output_dirs = os.path.join(output_dirs, "model", "sequentialnll", str(self.name))
        if not os.path.exists(self.output_dirs):
            os.makedirs(self.output_dirs)
        self.feature_name_mapping = {} 

    def clean_feature_names(self, feature_names: List[str]) -> List[str]:
        cleaned_names = []
        seen_names = {}
        for idx, name in enumerate(feature_names):
            cleaned = re.sub(r'[^a-zA-Z0-9_.]', '_', str(name))
            cleaned = re.sub(r'_+', '_', cleaned)
            cleaned = cleaned.strip('_')
            if not cleaned:
                cleaned = f'feature_{idx}'
            original_cleaned = cleaned
            counter = 0
            while cleaned in seen_names:
                counter += 1
                cleaned = f'{original_cleaned}_{counter}'
            seen_names[cleaned] = True
            cleaned_names.append(cleaned)
            self.feature_name_mapping[cleaned] = name
        return cleaned_names
    
    def prepare_data(self, df: pd.DataFrame, 
                    selected_features: List[str],
                    taget_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = df[selected_features].values
        y = df[taget_col].values
        dates = df['trade_time'].values

        logger.panel(f"  特征矩阵 X: {X.shape} (样本数 × 特征数) \n"
                     f"  目标变量 y: {y.shape}"
                     f"  时间序列: {len(dates)}\n"
                     f"\n  数据类型:\n"
                     f"    X: {X.dtype}\n"
                     f"    y: {y.dtype}\n"
                     f"\n  数据范围:\n"
                     f"    X最小值: {X.min():.6f}\n"
                     f"    X最大值: {X.max():.6f}\n"
                     f"    y最小值: {y.min():.6f}\n"
                     f"    y最大值: {y.max():.6f}",
                     "提取特征矩阵和目标变量")
        return X, y, dates
    
    def split_data(self, X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                   train_ratio: float = 0.7) -> Tuple:
        logger.panel(
            f"  ✓ 正确: 前70%训练，后30%校验\n"
            "  ✗ 错误: 随机划分（会导致用未来预测过去\n",
            title="时间序列预测必须按时间顺序划分数据"
        )
        split_idx = int(len(X) * train_ratio)

        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        dates_train, dates_val = dates[:split_idx], dates[split_idx:]

        logger.panel(
            f"  训练/校验比例: {train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%\n"
            f"  划分点索引: {split_idx}\n"
            f"\n  [训练集]\n"
            f"    样本数: {len(X_train):,}\n"
            f"    时间范围: {dates_train[0]} 至 {dates_train[-1]}\n"
            f"    时间跨度: {(pd.Timestamp(dates_train[-1]) - pd.Timestamp(dates_train[0])).days} 天\n"
            f"    目标变量统计:\n"
            f"      均值: {y_train.mean():.6f}\n"
            f"      标准差: {y_train.std():.6f}\n"
            f"      正收益比例: {(y_train > 0).mean()*100:.2f}%\n"
            f"\n  [校验集]\n"
            f"    样本数: {len(X_val):,}\n"
            f"    时间范围: {dates_val[0]} 至 {dates_val[-1]}\n"
            f"    时间跨度: {(pd.Timestamp(dates_val[-1]) - pd.Timestamp(dates_val[0])).days} 天\n"
            f"    目标变量统计:\n"
            f"      均值: {y_val.mean():.6f}\n"
            f"      标准差: {y_val.std():.6f}\n"
            f"      正收益比例: {(y_val > 0).mean()*100:.2f}%\n", title="数据集信息"
        )

        # 检查训练集和校验集分布差异
        mean_diff = abs(y_train.mean() - y_val.mean())
        std_ratio = y_val.std() / y_train.std() if y_train.std() != 0 else 0
        
        content = f"    均值差异: {mean_diff:.6f}\n"
        content += f"    标准差比: {std_ratio:.2f}\n"

        if std_ratio > 1.5 or std_ratio < 0.67:
            content+= f"    ⚠️  警告: 校验集波动性与训练集差异较大\n"
        else:
            content+= f"    ✓ 训练集和校验集分布相对一致\n"

        logger.panel(content=content, title="[分布一致性检查]")
        
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)

        return X_train, X_val, y_train, y_val, dates_train, dates_val
    
    def create_rolling_window_samples(self, data):
        num_timesteps, num_features = data.shape
        num_samples = num_timesteps - self.train_params['seq_len'] + 1
        shape = (num_samples, self.train_params['seq_len'], num_features)
        strides = (data.strides[0], data.strides[0], data.strides[1])
        samples = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
        samples = samples.astype(np.float32)
        return samples
    
    def create_train_data_loader(self, x_samples, y_samples, shuffle=False):
        dataset = TensorDataset(torch.from_numpy(x_samples), torch.from_numpy(y_samples))
        loader = DataLoader(dataset=dataset, batch_size=self.train_params['batch_size'], shuffle=shuffle)
        return loader
    
    def create_predict_data_loader(self, test_samples, shuffle=False):
        dataset = TensorDataset(torch.from_numpy(test_samples))
        test_loader = DataLoader(dataset, batch_size=self.train_params['batch_size'], shuffle=shuffle)
        return test_loader

    def check_model_capacity(self, total_params: int, trainable_params: int, train_samples: int):
        param_sample_ratio = trainable_params / train_samples
        seq_len = self.train_params['seq_len']
        enc_in = self.params['enc_in']
        effective_data_points = train_samples * seq_len * enc_in
        param_datapoint_ratio = trainable_params / effective_data_points

        content = f"  训练样本数: {train_samples:,}\n"
        content += f"  可训练参数: {trainable_params:,}\n"
        content += f"  参数/样本比: {param_sample_ratio:.2f}\n"
        content += f"  有效数据点: {effective_data_points:,} (样本×seq_len×features)\n"
        content += f"  参数/数据点比: {param_datapoint_ratio:.6f}\n\n"

        if param_sample_ratio > 10:
            status = "🚨 严重过参数化"
            risk = "极高"
            title = "⚠️  模型容量检查 - 严重警告"
        elif param_sample_ratio > 1:
            status = "❌ 过参数化"
            risk = "高"
            title = "⚠️  模型容量检查 - 警告"
        elif param_sample_ratio > 0.1:
            status = "⚠️ 需要正则化"
            risk = "中等"
            title = "ℹ️  模型容量检查 - 提示"
        else:
            status = "✅ 参数量合理"
            risk = "低"
            title = "✅ 模型容量检查 - 正常"
        
        content += f"  状态: {status}\n"
        content += f"  过拟合风险: {risk}\n"
        logger.panel(content, title=title)
    
    def validate_data(self, autocode_data):
        """数据校验函数"""
        logger.rule("数据校验")
    
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
        logger.panel(
            f"  最小值: {autocode_data[factor_cols].min().min():.6e}"
            f"  最大值: {autocode_data[factor_cols].max().max():.6e}"
            f"  均值: {autocode_data[factor_cols].mean().mean():.6e}"
            f"  标准差: {autocode_data[factor_cols].std().mean():.6e}", title="特征统计"
        )
    
        # 4. 检查目标变量
        target_col = [c for c in autocode_data.columns if c.startswith('nxt1_ret_')][0]
        logger.panel(
            f"  范围: [{autocode_data[target_col].min():.6f}, {autocode_data[target_col].max():.6f}]"
            f"  均值: {autocode_data[target_col].mean():.6e}"
            f"  标准差: {autocode_data[target_col].std():.6f}",
            title="目标变量 ({target_col})"
        )
    
        # 5. 检查时间连续性
        dates = pd.to_datetime(autocode_data['trade_time'])
        time_gaps = dates.diff()
        logger.panel(
            f"\n时间跨度: {dates.min()} 至 {dates.max()}"
            f"  总样本数: {len(autocode_data)}"
            f"  时间间隔中位数: {time_gaps.median()}",title="检查时间连续性"
        )
    
        # 6. 检查异常值
        from scipy import stats
        z_scores = np.abs(stats.zscore(autocode_data[factor_cols]))
        outliers = (z_scores > 5).sum().sum()
        if outliers > 0:
            logger.print(f"\n⚠️ 发现 {outliers} 个极端异常值 (|z-score| > 5)")
    
        return True

    def train_model(self, model_method, train_loader, val_loader):
        model = model_method(**self.params).to(self.train_params['device'])

        content = ""
        for key, value in self.params.items():
            content += f"    {key}: {value}\n"
        logger.panel(content, title="模型超参")

        content = ""
        for key, value in self.train_params.items():
            content += f"    {key}: {value}\n"
        logger.panel(content, title="训练参数")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_samples = len(train_loader.dataset)
        
        self.check_model_capacity(total_params, trainable_params, train_samples)
        
        logger.panel(content=f"  总参数: {total_params:,}\n"
                     f"  可训练参数: {trainable_params:,}",
                     title="参数说明")
        
        # 根据配置选择损失函数
        loss_func_name = self.train_params.get('loss_func', 'mse')
        if loss_func_name == 'mse':
            criterion = torch.nn.MSELoss()
        elif loss_func_name == 'gaussian_nll':
            criterion = gaussian_nll_loss
        else:
            raise ValueError(f"Unsupported loss function: {loss_func_name}")

        if 'weight_decay' in self.train_params:
            optimizer = optim.AdamW(model.parameters(), lr=self.train_params['learning_rate'], weight_decay=self.train_params['weight_decay'])
        else:
            optimizer = optim.AdamW(model.parameters(), lr=self.train_params['learning_rate'])

        # 添加 Learning Rate Warmup (P0 修复)
        # warmup_ratio 默认 0.1，即前 10% 步数进行 warmup
        warmup_ratio = self.train_params.get('warmup_ratio', 0.1)
        total_steps = len(train_loader) * self.train_params['epochs']
        warmup_steps = int(total_steps * warmup_ratio)

        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
            
        scheduler = LambdaLR(optimizer, lr_lambda)
        logger.print(f"  Warmup 配置: {warmup_steps} 步 (总步数 {total_steps} 的 {warmup_ratio*100:.0f}%)")
        
        best_val_loss = float('inf')
        patience_counter = 0

        logger.print(f"开始训练 {model_method.__name__} (Loss: {loss_func_name})...")

        for epoch in range(self.train_params['epochs']):
            model.train()
            total_loss = 0
            
            for i, (batch_inputs, batch_targets) in enumerate(train_loader):
                batch_inputs = batch_inputs.to(self.train_params['device'])
                batch_targets = batch_targets.to(self.train_params['device'])
                
                optimizer.zero_grad()
                _, _, outputs = model(batch_inputs)

                if loss_func_name == 'gaussian_nll':
                    # 期望输出形状为 [batch, 2] -> (mean, var)
                    if outputs.shape[-1] == 2:
                        pred_mean = outputs[:, 0]
                        pred_var = outputs[:, 1]
                        loss = criterion(
                            pred_mean=pred_mean, pred_var=pred_var,
                            target=batch_targets,
                            lambda_diversity=self.train_params['lambda_diversity'],
                            lambda_pred_mean=self.train_params.get('lambda_pred_mean', 0.0))
                    else:
                        # 如果模型输出形状不对，抛出异常
                        raise ValueError("Model output shape mismatch for Gaussian NLL. Expected [batch, 2].")
                else:
                    # MSE 逻辑
                    if outputs.shape != batch_targets.shape:
                        if outputs.shape[-1] == 1 and len(outputs.shape) > len(batch_targets.shape):
                            outputs = outputs.squeeze(-1)
                        elif len(batch_targets.shape) == 1 and len(outputs.shape) == 2:
                            batch_targets = batch_targets.unsqueeze(-1)
                    loss = criterion(outputs, batch_targets)
                
                loss.backward()
                max_grad_norm = self.train_params.get('max_grad_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # 验证过程
            model.eval()
            val_loss = 0
            val_mse = 0
            with torch.no_grad():
                for i, (batch_inputs, batch_targets) in enumerate(val_loader):
                    batch_inputs = batch_inputs.to(self.train_params['device'])
                    batch_targets = batch_targets.to(self.train_params['device'])
                    
                    _, _, outputs = model(batch_inputs)
                    
                    if loss_func_name == 'gaussian_nll':
                        if outputs.shape[-1] == 2:
                            pred_mean = outputs[:, 0]
                            pred_var = outputs[:, 1]
                            loss = criterion(pred_mean, pred_var, batch_targets,
                                    lambda_diversity=self.train_params.get('lambda_diversity', 0))
                            # 同时记录 MSE 以便对比
                            mse = F.mse_loss(pred_mean, batch_targets.view_as(pred_mean))
                            val_mse += mse.item()
                        else:
                             raise ValueError("Model output shape mismatch for Gaussian NLL.")
                    else:
                        if outputs.shape != batch_targets.shape:
                            if outputs.shape[-1] == 1 and len(outputs.shape) > len(batch_targets.shape):
                                outputs = outputs.squeeze(-1)
                            elif len(batch_targets.shape) == 1 and len(outputs.shape) == 2:
                                batch_targets = batch_targets.unsqueeze(-1)
                        loss = criterion(outputs, batch_targets)
                        val_mse += loss.item() # 对于 MSE Loss, val_loss 就是 MSE

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_mse = val_mse / len(val_loader)

            msg = f"Epoch [{epoch+1}/{self.train_params['epochs']}] Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            if loss_func_name == 'gaussian_nll':
                msg += f", Val MSE: {avg_val_mse:.6f}"
            logger.print(msg)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                filename = os.path.join(self.output_dirs, 'best_sequential_model.pth')
                torch.save(model.state_dict(), filename)
            else:
                patience_counter += 1

            if patience_counter >= self.train_params['patience']:
                logger.print("Early stopping triggered.")
                break

        logger.print("✅ {model_method.__name__} training complete.")

    
    def predict(self, model_method, data_loader):
        model_path = os.path.join(self.output_dirs, 'best_sequential_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.panel(f"  模型路径: {model_path}\n", title="加载模型进行预测")
        
        model = model_method(**self.params).to(self.train_params['device'])
        model.load_state_dict(torch.load(model_path, map_location=self.train_params['device']))
        model.eval()

        all_predictions = []
        all_variances = []
        all_targets = []

        with torch.no_grad():
            for batch_data in data_loader:
                # 检查数据加载器返回的是单个张量还是元组
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    # 包含 (inputs, targets)
                    batch_inputs = batch_data[0].to(self.train_params['device'])
                    batch_targets = batch_data[1]
                    all_targets.append(batch_targets.numpy())
                else:
                    # 只包含 inputs
                    batch_inputs = batch_data[0].to(self.train_params['device'])
                
                _, _, outputs = model(batch_inputs)
                
                if outputs.shape[-1] == 2:
                    # 双头输出处理
                    pred_mean = outputs[:, 0]
                    pred_var = outputs[:, 1]
                    all_predictions.append(pred_mean.cpu().numpy())
                    all_variances.append(pred_var.cpu().numpy())
                else:
                    # 单头输出处理
                    if outputs.shape[-1] == 1 and len(outputs.shape) > 1:
                        outputs = outputs.squeeze(-1)
                    all_predictions.append(outputs.cpu().numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        
        # 只有当有targets时才拼接
        if len(all_targets) > 0:
            targets = np.concatenate(all_targets, axis=0)
        else:
            targets = None
        
        if len(all_variances) > 0:
            variances = np.concatenate(all_variances, axis=0)
            return predictions, variances, targets, model
        else:
            return predictions, targets, model
