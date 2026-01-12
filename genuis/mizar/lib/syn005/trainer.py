import re,os,copy,pdb
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from lib import logger

class Trainer(object):
    def __init__(self, params: Dict = None, train_params: Dict = None, output_dirs:str = None, name=None):
        self.model = None
        self.train_params = train_params
        self.params = params
        self.name = name
        self.output_dirs = os.path.join(output_dirs, "model", "autoencode", str(self.name))
        if not os.path.exists(self.output_dirs):
            os.makedirs(self.output_dirs)
        self.feature_name_mapping = {}  # 存储原始特征名到清理后特征名的映射
        # 自相关惩罚系数（可选），默认关闭。可在 train_params 中设置 autocorr_beta。
        self.autocorr_beta = float(self.train_params.get("autocorr_beta", 0.0))

    @staticmethod
    def autocorr_penalty(h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        计算隐层时间步的 Lag-1 自相关惩罚。
        h: [batch, seq_len, dim]
        return: scalar tensor
        """
        if h.dim() != 3 or h.size(1) < 2:
            return h.new_tensor(0.0)
        x = h[:, 1:, :]
        y = h[:, :-1, :]
        x = x - x.mean(dim=(0, 1), keepdim=True)
        y = y - y.mean(dim=(0, 1), keepdim=True)
        num = (x * y).sum(dim=(0, 1))
        den = (x.pow(2).sum(dim=(0, 1)).sqrt() * y.pow(2).sum(dim=(0, 1)).sqrt()) + eps
        corr = num / den  # [dim]
        return (corr.pow(2)).mean()


    def clean_feature_names(self, feature_names: List[str]) -> List[str]:
        """
        清理特征名称，移除LightGBM不支持的特殊字符
        
        """
        cleaned_names = []
        seen_names = {}  # 用于跟踪已使用的清理后名称，避免重复
        
        for idx, name in enumerate(feature_names):
            # 替换所有特殊字符为下划线
            # 保留字母、数字、下划线、点号
            cleaned = re.sub(r'[^a-zA-Z0-9_.]', '_', str(name))
            # 移除连续的下划线
            cleaned = re.sub(r'_+', '_', cleaned)
            # 移除开头和结尾的下划线
            cleaned = cleaned.strip('_')
            # 如果清理后为空，使用默认名称
            if not cleaned:
                cleaned = f'feature_{idx}'
            
            # 处理重复的清理后名称
            original_cleaned = cleaned
            counter = 0
            while cleaned in seen_names:
                counter += 1
                cleaned = f'{original_cleaned}_{counter}'
            
            seen_names[cleaned] = True
            cleaned_names.append(cleaned)
            # 保存映射关系
            self.feature_name_mapping[cleaned] = name
        
        return cleaned_names
    
    def prepare_data(self, df: pd.DataFrame, 
                    selected_features: List[str],
                    taget_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 特征矩阵X、目标变量y、时间序列dates
        """


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
        """
        按时间顺序划分训练集和校验集
        
        【重要】时间序列预测必须按时间顺序划分，不能随机划分！
        
        参数:
            X: 特征矩阵
            y: 目标变量
            dates: 时间序列
            train_ratio: 训练集比例
        
        返回:
            Tuple: (X_train, X_val, y_train, y_val, dates_train, dates_val)
        """

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
        std_ratio = y_val.std() / y_train.std()
        
        content = f"    均值差异: {mean_diff:.6f}\n"
        content += f"    标准差比: {std_ratio:.2f}\n"

        if std_ratio > 1.5 or std_ratio < 0.67:
            content+= f"    ⚠️  警告: 校验集波动性与训练集差异较大\n"
        else:
            content+= f"    ✓ 训练集和校验集分布相对一致\n"

        logger.panel(
            content=content,title="[分布一致性检查]"
        )
        
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)

        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)

        return X_train, X_val, y_train, y_val, dates_train, dates_val
    
    def create_rolling_window_samples(self, data):
        """
        将连续的时间序列数据转换为重叠的滚动窗口样本。
    
        Args:
            data (np.array): 输入的特征数据，形状为 (num_timesteps, num_features)。
            seq_len (int): 每个样本的时间窗口长度。
        
        Returns:
            np.array: 样本数据，形状为 (num_samples, seq_len, num_features)。
        """
        num_timesteps, num_features = data.shape
        num_samples = num_timesteps - self.train_params['seq_len'] + 1

        # 使用 numpy 的 stride_tricks 高效创建滚动窗口，避免循环
        shape = (num_samples, self.train_params['seq_len'], num_features)
        strides = (data.strides[0], data.strides[0], data.strides[1])
        samples = np.lib.stride_tricks.as_strided(data,
                                              shape=shape,
                                              strides=strides)
        samples = samples.astype(np.float32)
        return samples


    def create_train_data_loader(self, x_samples,  y_samples, shuffle=False):
        dataset = TensorDataset(torch.from_numpy(x_samples),
                                  torch.from_numpy(y_samples))
        loader = DataLoader(dataset=dataset, batch_size=self.train_params['batch_size'],
                             shuffle=shuffle)
        return loader

    def create_predict_data_loader(self, test_samples, shuffle=False):
        dataset = TensorDataset(torch.from_numpy(test_samples))
        test_loader = DataLoader(dataset, batch_size=self.train_params['batch_size'], shuffle=shuffle)
        return test_loader

    def check_model_capacity(self, total_params: int, trainable_params: int, train_samples: int):
        # 计算参数/样本比
        param_sample_ratio = trainable_params / train_samples

        # 计算有效样本数（考虑序列长度和特征数）
        seq_len = self.train_params['seq_len']
        enc_in = self.params['enc_in']
        effective_data_points = train_samples * seq_len * enc_in

        # 计算有效参数/数据点比
        param_datapoint_ratio = trainable_params / effective_data_points

        # 构建报告内容
        content = f"  训练样本数: {train_samples:,}\n"
        content += f"  可训练参数: {trainable_params:,}\n"
        content += f"  参数/样本比: {param_sample_ratio:.2f}\n"
        content += f"  有效数据点: {effective_data_points:,} (样本×seq_len×features)\n"
        content += f"  参数/数据点比: {param_datapoint_ratio:.6f}\n\n"

        # 判断状态
        if param_sample_ratio > 10:
            status = "🚨 严重过参数化"
            risk = "极高"
            color = "red"
            recommendations = [
                f"1. 减小 d_model (当前: {self.params.get('d_model', 'N/A')}) → 建议: {max(32, self.params.get('d_model', 128) // 2)}",
                f"2. 减少 e_layers (当前: {self.params.get('e_layers', 'N/A')}) → 建议: {max(1, self.params.get('e_layers', 3) - 1)}",
                "3. 增加训练数据（扩展历史数据范围）",
                "4. 实施数据增强（噪声扰动、时间窗口滑动）",
                f"5. 增加 dropout (当前: {self.params.get('dropout', 'N/A')}) → 建议: 0.2-0.3"
            ]
        elif param_sample_ratio > 1:
            status = "❌ 过参数化"
            risk = "高"
            color = "yellow"
            recommendations = [
                f"1. 适当减小 d_model (当前: {self.params.get('d_model', 'N/A')})",
                "2. 增加正则化（dropout, weight decay）",
                "3. 考虑数据增强",
                "4. 密切监控过拟合（train loss vs val loss）"
            ]
        elif param_sample_ratio > 0.1:
            status = "⚠️ 需要正则化"
            risk = "中等"
            color = "yellow"
            recommendations = [
                "1. 确保使用 dropout 和 early stopping",
                "2. 监控验证集性能",
                "3. 如果出现过拟合，考虑减小模型或增加数据"
            ]
        else:
            status = "✅ 参数量合理"
            risk = "低"
            color = "green"
            recommendations = [
                "模型容量适中，可以正常训练",
                "继续监控训练过程，确保收敛"
            ]
        
        content += f"  状态: {status}\n"
        content += f"  过拟合风险: {risk}\n\n"
        content += "  建议:\n"
        for rec in recommendations:
            content += f"    {rec}\n"
        
        # 根据风险等级选择标题样式
        if risk == "极高":
            title = "⚠️  模型容量检查 - 严重警告"
        elif risk == "高":
            title = "⚠️  模型容量检查 - 警告"
        elif risk == "中等":
            title = "ℹ️  模型容量检查 - 提示"
        else:
            title = "✅ 模型容量检查 - 正常"
        
        logger.panel(content, title=title)
        
        # 如果严重过参数化，额外警告
        if param_sample_ratio > 10:
            logger.print("\n" + "="*80)
            logger.print("🚨 严重警告: 模型参数数量是训练样本的 {:.1f} 倍！".format(param_sample_ratio))
            logger.print("   这可能导致:")
            logger.print("   - 严重过拟合（模型记忆训练集而非学习规律）")
            logger.print("   - 高自相关（模型学会复制输入而非提取特征）")
            logger.print("   - 泛化能力差（在新数据上表现不佳）")
            logger.print("   强烈建议调整模型参数或增加训练数据！")
            logger.print("="*80 + "\n")
        

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
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # 计算训练样本数
        train_samples = len(train_loader.dataset)
        
        # 参数/样本比检查
        self.check_model_capacity(total_params, trainable_params, train_samples)
        
        logger.panel(content=f"  总参数: {total_params:,}\n"
                     f"  可训练参数: {trainable_params:,}",
                     title="参数说明")
        
        
        criterion = torch.nn.MSELoss()
        if 'weight_decay' in self.train_params:
            optimizer = optim.Adam(model.parameters(), lr=self.train_params['learning_rate'], weight_decay=self.train_params['weight_decay'])
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.train_params['learning_rate'])

        best_val_loss = float('inf')
        patience_counter = 0
        for epoch in range(self.train_params['epochs']):
            model.train()
            total_loss = 0
            for i, (batch_inputs, _) in enumerate(train_loader):
                # 将数据移动到指定设备
                batch_inputs = batch_inputs.to(self.train_params['device'])
                # 清零梯度
                optimizer.zero_grad()

                # 前向传播 (is_training=True 是模型内部的默认或需要手动设置)
                # 我们的 TemporiorientTransformer 实现中，训练时自动遮盖
                enc_out, dec_out, outputs = model(batch_inputs)
                
                recon_loss = criterion(outputs, batch_inputs)
                if self.autocorr_beta > 0:
                    ac_penalty = self.autocorr_penalty(enc_out)
                    loss = recon_loss + self.autocorr_beta * ac_penalty
                else:
                    loss = recon_loss

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                #if (i + 1) % 100 == 0:
                #    logger.print(
                #    f"Epoch [{epoch+1}/{self.train_params['epochs']}], Step [{i+1}/{len(train_loader)}], Avg Loss: {total_loss / (i+1):.6f}"
                #    )
            # -- 验证阶段 --
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i, (batch_inputs, _) in enumerate(val_loader):
                    batch_inputs = batch_inputs.to(self.train_params['device'])
                    enc_out, dec_out, outputs = model(
                        batch_inputs, masking_ratio=0)  # 重建任务，验证时也需要重建
                    recon_loss = criterion(outputs, batch_inputs)
                    if self.autocorr_beta > 0:
                        ac_penalty = self.autocorr_penalty(enc_out)
                        loss = recon_loss + self.autocorr_beta * ac_penalty
                    else:
                        loss = recon_loss
                    val_loss += loss.item()
                    if (i + 1) % 100 == 0:
                        logger.print(
                        f" Step [{i+1}/{len(val_loader)}], Avg Loss: {val_loss / (i+1):.6f}"
                    )

            avg_val_loss = val_loss / len(val_loader)

            # -- 早停逻辑 --
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                filename = os.path.join(
                    self.output_dirs, 'best_temporiorient_model.pth')
                torch.save(model.state_dict(), filename)
                #logger.print(
                #f"Validation loss improved. Saved best model to 'best_temporiorient_model.pth'"
            #)
            else:
                patience_counter += 1
                logger.print(
                f"Validation loss did not improve. Patience: {patience_counter}/{self.train_params['patience']}"
                )

            if patience_counter >= self.train_params['patience']:
                logger.print("Early stopping triggered.")
                break

            filename = os.path.join(
                self.output_dirs, "temporiorient_encoder_pretrained_{0}.pth".format(epoch + 1))
            torch.save(model.state_dict(), filename)
            logger.print(
            f"--- Epoch [{epoch+1}/{self.train_params['epochs']}] Complete --- Avg Reconstruction Loss: {total_loss / len(train_loader):.6f} ---")
        logger.print("✅ TemporiorientTransformer pre-training complete.")

    def predict(self, model_method, data_loader, multi_timestep_extraction=True, save_for_evaluation=True):
        """
        使用训练好的模型生成隐层特征
        
        参数:
            data_loader: PyTorch DataLoader
            model: 训练好的模型
            multi_timestep_extraction: 是否提取多个时间步的特征
        
        返回:
            np.ndarray: 隐层特征数组
        """
        model_path = os.path.join(self.output_dirs, 'best_temporiorient_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}\n请先运行 train_model()")

        logger.panel(f"  模型路径: {model_path}\n", title="加载模型")
        params = copy.deepcopy(self.params)
        del params['masking_ratio']
        model = model_method(
            **params).to(self.train_params['device'])
        model.load_state_dict(torch.load(model_path, map_location=self.train_params['device']))
        model.eval()
        
        all_factors = []
        all_original = []       # 新增
        all_reconstructed = []
        
        logger.panel(
            f"  多时间步提取: {multi_timestep_extraction}\n"
            f"  隐层维度: {self.train_params.get('d_model', 256)}\n",
            title="开始生成隐层特征"
        )
        
        if multi_timestep_extraction:
            # 提取多个时间步的特征
            timesteps_to_extract = [-1, -3, -10]  # 最近、短期、中期
            logger.print(f"  提取时间步: {timesteps_to_extract}")
            
            with torch.no_grad():
                for (batch_inputs,) in data_loader:
                    batch_inputs = batch_inputs.to(self.train_params['device'])
                    enc_out, dec_out, outputs  = model(batch_inputs, masking_ratio=0.0)
                    
                    # 提取多个时间步并拼接
                    multi_timestep_features = [enc_out[:, ts, :] for ts in timesteps_to_extract]
                    
                    # 拼接: [batch, len(timesteps) * d_model]
                    final_factors = torch.cat(multi_timestep_features, dim=1)
                    all_factors.append(final_factors.cpu().numpy())

                    if save_for_evaluation:
                        all_original.append(batch_inputs.cpu().numpy())
                        all_reconstructed.append(outputs.cpu().numpy())
            
            factor_dim = len(timesteps_to_extract) * self.params.get('d_model', 256)
            logger.print(f"  因子维度: {factor_dim} (={len(timesteps_to_extract)} × {self.params.get('d_model', 256)})")
        else:
            # 仅提取最后一个时间步
            with torch.no_grad():
                for (batch_inputs,) in data_loader:
                    batch_inputs = batch_inputs.to(self.train_params['device'])
                    enc_out, dec_out, outputs = model(batch_inputs, masking_ratio=0.0)
                    
                    # 提取最后一个时间点的因子
                    final_factors = enc_out[:, -1, :]  # Shape: [batch_size, d_model]
                    all_factors.append(final_factors.cpu().numpy())

                    if save_for_evaluation:
                        all_original.append(batch_inputs.cpu().numpy())
                        all_reconstructed.append(outputs.cpu().numpy())
            
            factor_dim = self.params.get('d_model', 256)
            logger.print(f"  因子维度: {factor_dim}")
        
        # 整合结果
        factors_array = np.concatenate(all_factors, axis=0)
        if save_for_evaluation:
            original_array = np.concatenate(all_original, axis=0)
            reconstructed_array = np.concatenate(all_reconstructed, axis=0)
        else:
            original_array = None
            reconstructed_array = None
            

        logger.panel(
            f"  生成的因子数组形状: {factors_array.shape}\n"
            f"  因子数值范围: [{factors_array.min():.6f}, {factors_array.max():.6f}]\n",
            title="✅ 隐层特征生成完成"
        )
        
        return factors_array, original_array, reconstructed_array