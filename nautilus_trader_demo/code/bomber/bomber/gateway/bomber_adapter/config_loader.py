"""
配置文件加载器

支持从 JSON 配置文件加载回测参数。
"""
import json
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """配置文件加载器"""

    def __init__(self, config_path: str = None):
        """
        初始化配置加载器

        参数:
            config_path: 配置文件路径（JSON 格式）
        """
        self.config_path = config_path
        self.config = {}

        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件

        参数:
            config_path: 配置文件路径

        返回:
            配置字典
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 只支持 JSON 格式
        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {path.suffix}，请使用 .json")

        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        参数:
            key: 配置键
            default: 默认值

        返回:
            配置值
        """
        return self.config.get(key, default)

    def get_nested(self, *keys, default: Any = None) -> Any:
        """
        获取嵌套配置项

        参数:
            *keys: 配置键路径
            default: 默认值

        返回:
            配置值
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default

            if value is None:
                return default

        return value

    def merge(self, other_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置（其他配置覆盖当前配置）

        参数:
            other_config: 其他配置字典

        返回:
            合并后的配置字典
        """
        merged = self.config.copy()
        merged.update(other_config)
        return merged

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        返回:
            配置字典
        """
        return self.config.copy()

    def save(self, config_path: str):
        """
        保存配置到文件

        参数:
            config_path: 配置文件路径
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 只支持 JSON 格式
        if path.suffix == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {path.suffix}，请使用 .json")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    便捷函数：加载配置文件

    参数:
        config_path: 配置文件路径

    返回:
        配置字典
    """
    loader = ConfigLoader(config_path)
    return loader.to_dict()


def create_default_config(config_path: str):
    """
    创建默认配置文件

    参数:
        config_path: 配置文件路径
    """
    default_config = {
        'strategy_id': 'default_strategy',
        'strategy_ver': 'v1.0',
        'output_type': 'BACKTEST_RESULT',
        'period_start': '2024-01-01',
        'period_end': '2024-12-31',
        'account': {
            'initial_capital': 10000000,
            'currency': 'CNY'
        },
        'execution': {
            'order_type': 'MARKET',
            'execution_price': 'close'
        },
        'cost': {
            'commission_bps': 2.3,
            'impact_bps': 10
        },
        'venue': {
            'name': 'CFFEX',
            'oms_type': 'NETTING'
        },
        'fill_model': {
            'path': 'bomber.backtest.models.fill:ProbabilisticFillModel',
            'config_path': 'bomber.backtest.config:FillModelConfig',
            'config': {
                'prob_fill_on_limit': 0.9,
                'prob_slippage': 0.1,
                'random_seed': 42
            }
        },
        'matching': {
            'bar_execution': True,
            'trade_execution': True,
            'bar_adaptive_high_low_ordering': False
        }
    }

    loader = ConfigLoader()
    loader.config = default_config
    loader.save(config_path)
