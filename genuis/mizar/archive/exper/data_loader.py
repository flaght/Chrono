"""
数据加载模块

负责从各种数据源加载数据，支持CSV、Parquet、Feather等格式。
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Union
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 尝试导入项目特定的模块（如果存在）
try:
    from lib.lsx001 import fetch_times
    from kdutils.macro2 import base_path
    HAS_PROJECT_MODULES = True
except ImportError:
    HAS_PROJECT_MODULES = False
    print("警告: 未找到项目特定模块，将使用通用数据加载方式")


class DataLoader:
    """
    数据加载器类
    
    支持多种数据加载方式：
    1. 从CSV文件加载
    2. 从Parquet文件加载
    3. 从Feather文件加载
    4. 从项目特定路径加载（如果可用）
    5. 生成模拟数据（用于演示）
    """
    
    def __init__(self, use_mock_data: bool = False):
        """
        初始化数据加载器
        
        参数:
            use_mock_data: 是否使用模拟数据（默认False）
        """
        self.use_mock_data = use_mock_data
    
    def load_from_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        从CSV文件加载数据
        
        参数:
            file_path: CSV文件路径
            **kwargs: 传递给pd.read_csv的其他参数
        
        返回:
            DataFrame: 加载的数据
        """
        print(f"从CSV文件加载数据: {file_path}")
        df = pd.read_csv(file_path, **kwargs)
        print(f"✓ 数据加载成功: {df.shape}")
        return df
    
    def load_from_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        从Parquet文件加载数据（推荐，速度更快）
        
        参数:
            file_path: Parquet文件路径
            **kwargs: 传递给pd.read_parquet的其他参数
        
        返回:
            DataFrame: 加载的数据
        """
        print(f"从Parquet文件加载数据: {file_path}")
        df = pd.read_parquet(file_path, **kwargs)
        print(f"✓ 数据加载成功: {df.shape}")
        return df
    
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
    
    def load_from_project(self, method: str, task_id: int, 
                         instruments: str, period: int, name: str) -> pd.DataFrame:
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
        if not HAS_PROJECT_MODULES:
            raise ImportError("项目特定模块不可用，无法使用此方法")
        
        # 获取时间数组
        time_array = fetch_times(
            method=method,
            task_id=task_id,
            instruments=instruments
        )
        
        # 构建文件路径
        dirs = os.path.join(
            base_path, method, instruments, 'temp', "model",
            str(task_id), str(period)
        )
        filename = os.path.join(dirs, f"{name}_data.feather")
        
        print(f"从项目路径加载数据: {filename}")
        df = pd.read_feather(filename)
        print(f"✓ 数据加载成功: {df.shape}")
        return df
    
    def generate_mock_data(self, n_samples: int = 10000, 
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
        dates = pd.date_range('2022-07-25 09:30:00', periods=n_samples, freq='15min')
        
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
    
    def load(self, source: Optional[Union[str, dict]] = None, **kwargs) -> pd.DataFrame:
        """
        统一的数据加载接口
        
        参数:
            source: 数据源
                - 如果是字符串: 作为文件路径，根据扩展名自动判断格式
                - 如果是字典: 包含加载参数（method, task_id等）
                - 如果是None且use_mock_data=True: 生成模拟数据
            **kwargs: 其他参数
        
        返回:
            DataFrame: 加载的数据
        """
        # 如果使用模拟数据
        if self.use_mock_data:
            return self.generate_mock_data(**kwargs)
        
        # 如果source是字典，使用项目特定加载方式
        if isinstance(source, dict):
            return self.load_from_project(**source)
        
        # 如果source是文件路径
        if isinstance(source, str):
            file_ext = os.path.splitext(source)[1].lower()
            
            if file_ext == '.csv':
                return self.load_from_csv(source, **kwargs)
            elif file_ext == '.parquet':
                return self.load_from_parquet(source, **kwargs)
            elif file_ext == '.feather':
                return self.load_from_feather(source, **kwargs)
            else:
                raise ValueError(f"不支持的文件格式: {file_ext}")
        
        # 如果source为None，尝试从环境变量或默认路径加载
        raise ValueError("请提供数据源（文件路径或加载参数）")
    
    def validate_data(self, df: pd.DataFrame, target_col: str = 'nxt1_ret_15h') -> bool:
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
            print(f"✗ 缺少必需的列: {missing_cols}")
            return False
        
        # 检查数据类型
        if not pd.api.types.is_datetime64_any_dtype(df['trade_time']):
            print("⚠️  警告: trade_time不是日期时间类型，将尝试转换")
            df['trade_time'] = pd.to_datetime(df['trade_time'])
        
        # 检查数据形状
        if len(df) == 0:
            print("✗ 数据为空")
            return False
        
        print(f"✓ 数据验证通过")
        print(f"  数据形状: {df.shape}")
        print(f"  时间范围: {df['trade_time'].min()} 至 {df['trade_time'].max()}")
        print(f"  内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return True

