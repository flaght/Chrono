import pandas as pd
import numpy as np
import os, pdb
from typing import Optional, Union

from lib.lsx001 import fetch_times
from kdutils.macro2 import base_path
from lib import logger



class DataLoader(object):

    def __init__(self, use_mock_data: bool = False):
        """
        初始化数据加载器
        
        参数:
            use_mock_data: 是否使用模拟数据（默认False）
        """
        self.use_mock_data = use_mock_data

    def generate_mock_data(self,
                           n_samples: int = 10000,
                           n_features: int = 300,
                           seed: int = 42) -> pd.DataFrame:
        """
        生成模拟数据（仅用于演示和测试）
        
        参数:
            n_samples: 样本数量
            n_features: 特征数量
            seed: 随机种子
        
        返回:
            DataFrame: 模拟数据
        """
        print("⚠️  生成模拟数据（仅用于演示）")
        print(f"  样本数: {n_samples:,}")
        print(f"  特征数: {n_features}")

        np.random.seed(seed)

        # 生成模拟时间序列（15分钟间隔）
        dates = pd.date_range('2022-07-25 09:30:00',
                              periods=n_samples,
                              freq='15min')

        # 生成模拟特征
        feature_data = np.random.randn(n_samples, n_features) * 0.5
        feature_names = [f'factor_{i}' for i in range(n_features)]

        # 生成模拟目标变量（带一些真实的相关性）
        target = np.random.randn(n_samples) * 0.01
        for i in range(5):
            target += feature_data[:, i] * 0.002

        # 组装DataFrame
        df = pd.DataFrame(feature_data, columns=feature_names)
        df.insert(0, 'trade_time', dates)
        df.insert(1, 'code', 'IM')
        df['nxt1_ret_15h'] = target

        # 随机插入一些NaN（模拟真实数据的缺失值）
        mask = np.random.random(df.shape) < 0.05
        df = df.mask(mask)

        print("✓ 模拟数据生成完成")
        return df

    def load_from_project(self, method: str, task_id: int, instruments: str,
                          period: int, name: str) -> pd.DataFrame:
        """
        从项目特定路径加载数据（如果项目模块可用）
        
        参数:
            method: 方法名
            task_id: 任务ID
            instruments: 合约代码
            period: 周期
            name: 数据名称
        
        返回:
            DataFrame: 加载的数据
        """

        # 获取时间数组
        time_array = fetch_times(method=method,
                                 task_id=task_id,
                                 instruments=instruments)

        pdb.set_trace()
        # 构建文件路径
        dirs = os.path.join(base_path, method, instruments, 'temp', "model",
                            str(task_id), str(period))
        filename = os.path.join(dirs, f"{name}_data.feather")

        logger.print(f"从项目路径加载数据: {filename}")
        df = pd.read_feather(filename)
        logger.print(f"✓ 数据加载成功: {df.shape}")

        #df = df[(df.trade_time >= time_array['train_time'][0])&(df.trade_time <= time_array['val_time'][1])]
        train_data = df[(df.trade_time >= time_array['train_time'][0])&(df.trade_time <= time_array['train_time'][1])]
        val_data = df[(df.trade_time >= time_array['val_time'][0])&(df.trade_time <= time_array['val_time'][1])]
        test_data = df[(df.trade_time >= time_array['test_time'][0])&(df.trade_time <= time_array['test_time'][1])]
        return train_data,val_data,test_data

    def load_from_feather(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        从Feather文件加载数据
        
        参数:
            file_path: Feather文件路径
            **kwargs: 传递给pd.read_feather的其他参数
        
        返回:
            DataFrame: 加载的数据
        """
        print(f"从Feather文件加载数据: {file_path}")
        df = pd.read_feather(file_path, **kwargs)
        print(f"✓ 数据加载成功: {df.shape}")
        return df

    def validate_data(self,
                      df: pd.DataFrame,
                      target_col: str = 'nxt1_ret_15h') -> bool:
        """
        验证数据的基本要求
        
        参数:
            df: 数据框
            target_col: 目标变量列名
        
        返回:
            bool: 验证是否通过
        """
        print("\n验证数据格式...")

        # 检查必需的列
        required_cols = ['trade_time', 'code', target_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"✗ 缺少必需的列: {missing_cols}")
            return False

        # 检查数据类型
        if not pd.api.types.is_datetime64_any_dtype(df['trade_time']):
            logger.warning("⚠️  警告: trade_time不是日期时间类型，将尝试转换")
            df['trade_time'] = pd.to_datetime(df['trade_time'])

        # 检查数据形状
        if len(df) == 0:
            logger.error("✗ 数据为空")
            return False

        logger.info(f"✓ 数据验证通过")
        logger.info(f"  数据形状: {df.shape}")
        logger.info(f"  时间范围: {df['trade_time'].min()} 至 {df['trade_time'].max()}")
        logger.info(f"  内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        return True
