"""
BacktestEnv - 回测环境对象
模拟生产系统的 rp.Start(conf) 返回的 env 对象。
策略通过 env 访问回测平台能力（持仓查询、数据加载等）。
"""
import os
import xml.etree.ElementTree as ET
from bomber_adapter.data_api import DataAPI


class BacktestEnv:
    def __init__(self, config_path="", data_dir=None, ddb_config=None, **params):
        self._config = {}
        self._params = params
        env_data_dir = os.environ.get("BOMBER_DATA_DIR") or os.environ.get("BT_DATA_DIR")
        self._data_dir = data_dir or env_data_dir or "./data"
        self._ddb_config = ddb_config
        self._bridge = None  # 由 BomberBridge.on_start 注入

        # 初始化数据接口
        self.data_api = DataAPI(
            data_dir=self._data_dir,
            ddb_config=ddb_config
        )

        if config_path and os.path.exists(config_path):
            try:
                tree = ET.parse(config_path)
                for elem in tree.getroot().iter():
                    if elem.text and elem.text.strip():
                        self._config[elem.tag] = elem.text.strip()
            except Exception:
                pass

    def set_bridge(self, bridge):
        """由 BomberBridge.on_start 调用，注入桥接器引用"""
        self._bridge = bridge

    def get_position(self, code: str) -> float:
        """
        查询当前持仓（从 Nautilus 引擎实时获取）

        返回:
            float: 净持仓量（正=多，负=空，0=空仓）
        """
        if self._bridge is not None:
            return self._bridge.get_position(code)
        return 0.0

    def get_positions(self) -> dict:
        """
        获取所有当前持仓信息（从 Nautilus 引擎实时获取）

        返回:
            dict: {合约代码: 持仓数量}，正数表示多头，负数表示空头
        """
        if self._bridge is not None:
            return self._bridge.get_positions()
        return {}

    def get_param(self, key, default=None):
        return self._config.get(key, self._params.get(key, default))

    def get_trading_date(self):
        return self._params.get("trading_date", "")

    @property
    def is_backtest(self):
        return True


def Start(conf="", data_dir=None, ddb_config=None, **kwargs):
    """
    对应生产的 rp.Start(conf)

    参数:
        conf: 配置文件路径
        data_dir: 数据目录（可选）
        ddb_config: DolphinDB 配置（可选）
    """
    return BacktestEnv(conf, data_dir=data_dir, ddb_config=ddb_config, **kwargs)
