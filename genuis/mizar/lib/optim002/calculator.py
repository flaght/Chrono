import importlib
import pdb
from matplotlib.pyplot import cla
import pandas as pd
from typing import Dict, List, Any, Tuple
from lumina.impulse.base import ImpulseBase

format_mapping = {
    1: {
        'format': 'default_key0',
        'param_names': [],
            'param_types': [],
            'default_keys': [()]
        },
    2: {
        'format': 'default_keys0',
        'param_names': ['window', 'ewm'],
        'param_types': ['int', 'categorical'],
        'default_keys': [(5, 1), (10, 1), (5, 0), (10, 0)]
    },
    3: {
        'format': 'default_keys1',
        'param_names': ['window', 'weriod', 'ewm'],
        'param_types': ['int', 'int', 'categorical'],
        'default_keys': [(5, 10, 1), (10, 15, 1), (5, 10, 0), (10, 15, 0)]
    },
    4: {
        'format': 'default_keys2',
        'param_names': ['window', 'fast', 'slow', 'ewm'],
        'param_types': ['int', 'int', 'int', 'categorical'],
        'default_keys': [(5, 5, 10, 1), (10, 10, 15, 1), (5, 5, 10, 0), (10, 10, 15, 0)]
    },
    5: {
        'format': 'default_keys3',
        'param_names': ['window', 'fast', 'slow', 'weriod', 'ewm'],
        'param_types': ['int', 'int', 'int', 'int', 'categorical'],
        'default_keys': [(5, 5, 10, 10, 1), (10, 5, 10, 15, 1), (5, 5, 10, 10, 0), (10, 5, 10, 15, 0)]
    },
    6: {
        'format': 'default_keys8',
        'param_names': ['window', 'fast', 'medium', 'slow', 'weriod', 'ewm'],
        'param_types': ['int', 'int', 'int', 'int', 'int', 'categorical'],
        'default_keys': [(5, 5, 10, 15, 10, 1), (10, 5, 10, 15, 15, 1), (5, 5, 10, 15, 10, 0), (10, 5, 10, 15, 15, 0)]
    }
        }

class ImpulseCalculator(object):
    def __init__(self, impulse_version: str = 'i017'):
        
        self.impulse_version = impulse_version
        #动态模块
        self.impulse_module = importlib.import_module(
            f'lumina.impulse.{impulse_version}')

    
    def get_class(self, name):
        factor_name = "Impulse{0}".format(name.capitalize())
        if hasattr(self.impulse_module, factor_name):
            factor_class = getattr(self.impulse_module, factor_name)
            try:
                if issubclass(factor_class, ImpulseBase):
                    return factor_class
            except ImportError:
                return None


    def detect_parameter_format(self, factor_class) -> Dict[str, Any]:
        temp_instance = factor_class()
        keys_attrs = [attr for attr in dir(temp_instance) if attr.endswith('_keys')]
        keys_attr = keys_attrs[0]  # 通常只有一个keys属性
        keys_value = getattr(temp_instance, keys_attr)
        # 获取一个样本参数来判断格式
        if hasattr(keys_value, '__iter__') and len(keys_value) > 0:
            sample_params = list(keys_value)[0]
        
        param_length = len(sample_params)

        detected_format = format_mapping[param_length]
        
        # 使用实际的keys值
        if hasattr(keys_value, '__iter__'):
            detected_format['default_keys'] = list(keys_value)
        else:
            # 如果keys_value不是可迭代的，使用默认值
            pass

        return detected_format

    def calculate_with_class(self, factor_class, params: Tuple,
                                     market_data: pd.DataFrame) -> pd.Series:
        try:
            custom_keys = params
            factor_instance = factor_class(keys=custom_keys)
            result_dict = factor_instance.calc_impulse(market_data)
            return result_dict
        except Exception as e:
            print(f"Error calculating factor {factor_class.__name__} with params {params}: {e}")
            return pd.Series(dtype=float)
    
    def calculate_with_name(self, name, params: Tuple,
                                     market_data: pd.DataFrame) -> pd.Series:
        try:
            factor_class = self.get_class(name=name)
            custom_keys = params
            factor_instance = factor_class(keys=custom_keys)
            result_dict = factor_instance.calc_impulse(market_data)
            return result_dict
        except Exception as e:
            print(f"Error calculating factor {factor_class.__name__} with params {params}: {e}")
            return pd.Series(dtype=float)