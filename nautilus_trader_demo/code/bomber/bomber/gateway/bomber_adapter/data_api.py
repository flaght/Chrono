"""
DataAPI - 策略数据加载接口
为策略提供统一的数据加载接口，支持加载各类辅助数据。
"""
import pandas as pd
import os
from typing import List, Dict, Optional
from datetime import datetime


class DataAPI:
    """
    策略数据加载接口

    策略可以在 initialize() 方法中调用此接口加载所需数据。

    使用示例：
        class MyStrategy(rp.IPyStrategy):
            def initialize(self, name, env):
                super().initialize(name, env)

                # 加载期货公司持仓数据
                self.position_data = env.data_api.load_position_data(
                    inst='IC2406',
                    start_date='2024-01-01',
                    end_date='2024-06-30'
                )

                # 加载期权 Greeks 数据
                self.greeks_data = env.data_api.load_greeks_data(
                    inst='IO2406-C-5000',
                    start_date='2024-01-01',
                    end_date='2024-06-30'
                )
    """

    def __init__(self, data_dir: str, ddb_config: dict = None):
        """
        初始化数据接口

        参数:
            data_dir: 数据目录
            ddb_config: DolphinDB 配置（可选）
        """
        self._data_dir = data_dir
        self._ddb_config = ddb_config
        self._ddb_session = None

    @staticmethod
    def _escape_sql(value: str) -> str:
        """转义 SQL 字符串值中的单引号"""
        return value.replace("'", "''")

    def _get_ddb_session(self):
        """获取 DolphinDB 会话"""
        if self._ddb_session is None and self._ddb_config:
            try:
                import dolphindb as ddb
                self._ddb_session = ddb.Session()
                self._ddb_session.connect(
                    self._ddb_config['host'],
                    self._ddb_config['port'],
                    self._ddb_config['userid'],
                    self._ddb_config['password']
                )
            except Exception as e:
                print(f"[DataAPI] 连接 DolphinDB 失败: {e}")
                self._ddb_session = None
        return self._ddb_session

    def load_position_data(self, inst: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载期货公司持仓数据

        参数:
            inst: 合约代码（如 'IC2406'）
            start_date: 开始日期（'YYYY-MM-DD'）
            end_date: 结束日期（'YYYY-MM-DD'）

        返回:
            DataFrame: 持仓数据，包含以下列：
                - date: 日期
                - inst: 合约代码
                - company: 期货公司名称
                - long_position: 多头持仓
                - short_position: 空头持仓
                - net_position: 净持仓
        """
        # 尝试从本地文件加载
        local_file = os.path.join(self._data_dir, 'position', f'{inst}_position.csv')
        if os.path.exists(local_file):
            df = pd.read_csv(local_file)
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            return df[mask]

        # 尝试从 DolphinDB 加载
        session = self._get_ddb_session()
        if session:
            try:
                safe_inst = self._escape_sql(inst)
                df = session.run(f"""
                    select tradeDate as date, instrumentId as inst,
                           brokerName as company,
                           longPosition as long_position,
                           shortPosition as short_position,
                           (longPosition - shortPosition) as net_position
                    from position_data
                    where instrumentId = '{safe_inst}'
                      and tradeDate >= {start_date.replace('-', '.')}
                      and tradeDate <= {end_date.replace('-', '.')}
                """)
                return df
            except Exception as e:
                print(f"[DataAPI] 从 DolphinDB 加载持仓数据失败: {e}")

        # 返回空 DataFrame
        print(f"[DataAPI] 未找到持仓数据: {inst}")
        return pd.DataFrame(columns=['date', 'inst', 'company', 'long_position', 'short_position', 'net_position'])

    def load_greeks_data(self, inst: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载期权 Greeks 数据

        参数:
            inst: 合约代码（如 'IO2406-C-5000'）
            start_date: 开始日期（'YYYY-MM-DD'）
            end_date: 结束日期（'YYYY-MM-DD'）

        返回:
            DataFrame: Greeks 数据，包含以下列：
                - date: 日期
                - inst: 合约代码
                - delta: Delta
                - gamma: Gamma
                - theta: Theta
                - vega: Vega
                - rho: Rho
                - implied_volatility: 隐含波动率
        """
        # 尝试从本地文件加载
        local_file = os.path.join(self._data_dir, 'greeks', f'{inst}_greeks.csv')
        if os.path.exists(local_file):
            df = pd.read_csv(local_file)
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            return df[mask]

        # 尝试从 DolphinDB 加载
        session = self._get_ddb_session()
        if session:
            try:
                safe_inst = self._escape_sql(inst)
                df = session.run(f"""
                    select tradeDate as date, instrumentId as inst,
                           delta, gamma, theta, vega, rho,
                           impliedVolatility as implied_volatility
                    from options_greeks
                    where instrumentId = '{safe_inst}'
                      and tradeDate >= {start_date.replace('-', '.')}
                      and tradeDate <= {end_date.replace('-', '.')}
                """)
                return df
            except Exception as e:
                print(f"[DataAPI] 从 DolphinDB 加载 Greeks 数据失败: {e}")

        # 返回空 DataFrame
        print(f"[DataAPI] 未找到 Greeks 数据: {inst}")
        return pd.DataFrame(columns=['date', 'inst', 'delta', 'gamma', 'theta', 'vega', 'rho', 'implied_volatility'])

    def load_custom_data(self, table_name: str, start_date: str = None,
                        end_date: str = None, filters: Dict = None) -> pd.DataFrame:
        """
        加载自定义数据

        参数:
            table_name: 表名（本地文件名为 table_name.csv）
            start_date: 开始日期（可选）
            end_date: 结束日期（可选）
            filters: 过滤条件（可选），如 {'inst': 'IC2406'}

        返回:
            DataFrame: 数据表
        """
        # 尝试从本地文件加载
        local_file = os.path.join(self._data_dir, 'custom', f'{table_name}.csv')
        if os.path.exists(local_file):
            df = pd.read_csv(local_file)

            # 应用日期过滤
            if 'date' in df.columns and start_date and end_date:
                df['date'] = pd.to_datetime(df['date'])
                mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                df = df[mask]

            # 应用其他过滤条件
            if filters:
                for key, value in filters.items():
                    if key in df.columns:
                        df = df[df[key] == value]

            return df

        # 尝试从 DolphinDB 加载
        session = self._get_ddb_session()
        if session:
            try:
                query = f"select * from {table_name}"
                conditions = []

                # 缓存 schema，避免重复查询
                col_names = session.run(f"schema({table_name})")['colDefs']['name'].values
                has_trade_date = 'tradeDate' in col_names
                if start_date and has_trade_date:
                    conditions.append(f"tradeDate >= {start_date.replace('-', '.')}")
                if end_date and has_trade_date:
                    conditions.append(f"tradeDate <= {end_date.replace('-', '.')}")

                if filters:
                    for key, value in filters.items():
                        safe_value = self._escape_sql(str(value))
                        conditions.append(f"{key} = '{safe_value}'")

                if conditions:
                    query += " where " + " and ".join(conditions)

                df = session.run(query)
                return df
            except Exception as e:
                print(f"[DataAPI] 从 DolphinDB 加载自定义数据失败: {e}")

        # 返回空 DataFrame
        print(f"[DataAPI] 未找到自定义数据: {table_name}")
        return pd.DataFrame()

    def load_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载交易日历

        参数:
            start_date: 开始日期
            end_date: 结束日期

        返回:
            DataFrame: 交易日历，包含以下列：
                - date: 日期
                - is_trading_day: 是否为交易日
        """
        # 尝试从本地文件加载
        local_file = os.path.join(self._data_dir, 'calendar.csv')
        if os.path.exists(local_file):
            df = pd.read_csv(local_file)
            df['date'] = pd.to_datetime(df['date'])
            mask = (df['date'] >= start_date) & (df['date'] <= end_date)
            return df[mask]

        # 尝试从 DolphinDB 加载
        session = self._get_ddb_session()
        if session:
            try:
                df = session.run(f"""
                    select calendarDate as date, isTradingDay as is_trading_day
                    from trading_calendar
                    where calendarDate >= {start_date.replace('-', '.')}
                      and calendarDate <= {end_date.replace('-', '.')}
                """)
                return df
            except Exception as e:
                print(f"[DataAPI] 从 DolphinDB 加载交易日历失败: {e}")

        # 返回空 DataFrame
        print(f"[DataAPI] 未找到交易日历")
        return pd.DataFrame(columns=['date', 'is_trading_day'])

    def get_contract_info(self, code: str = None, product: str = None) -> pd.DataFrame:
        """查询合约主数据（乘数、交易所、标的指数等）。"""
        pq_path = os.path.join(self._data_dir, "contract_master.parquet")
        if not os.path.exists(pq_path):
            return pd.DataFrame()
        df = pd.read_parquet(pq_path)
        if code:
            df = df[df["code"] == code]
        if product:
            df = df[df["product"] == product]
        return df

    def get_index_map(self, product: str = None) -> pd.DataFrame:
        """查询品种对应的标的指数。"""
        pq_path = os.path.join(self._data_dir, "index_map.parquet")
        if not os.path.exists(pq_path):
            return pd.DataFrame()
        df = pd.read_parquet(pq_path)
        if product:
            df = df[df["product"] == product]
        return df

    def load_continuing_multiplier(self, date: str, product_id: str, algo_id: int = 1) -> dict:
        """
        加载指定日期的连续乘数

        参数:
            date: 交易日期（'YYYY-MM-DD'）
            product_id: 品种 ID（如 'IM', 'IC', 'IF'）
            algo_id: 策略 ID（用于选择不同版本的乘数计算方式）

        返回:
            dict: {
                'main_contract': {'code': 'IM2609', 'multiplier_1min': 1.466, 'multiplier_5min': 1.466},
                'sub_contract': {'code': 'IM2612', 'multiplier_1min': 1.441, 'multiplier_5min': 1.441}
            }
        """
        # 从按品种拆分的 Parquet 文件加载
        pq_file = os.path.join(self._data_dir, 'continuing_multiplier', f'{product_id}.parquet')
        if not os.path.exists(pq_file):
            print(f"[DataAPI] 未找到连续乘数文件: {pq_file}")
            return {}

        try:
            # 只加载需要的列，减少内存使用
            df = pd.read_parquet(pq_file, columns=['TradeDate', 'productID', 'algoID', 'data_type', 'Code', 'multiplier'])
            df['TradeDate'] = pd.to_datetime(df['TradeDate'])
            target_date = pd.to_datetime(date)

            # 筛选指定日期和 algoID 的数据
            df_date = df[(df['TradeDate'] == target_date) & (df['algoID'] == algo_id)]
            if df_date.empty:
                print(f"[DataAPI] 未找到日期 {date} 品种 {product_id} algoID={algo_id} 的连续乘数")
                return {}

            result = {}

            # 主力合约（data_type 以 'main_' 开头）
            main_data = df_date[df_date['data_type'].str.startswith('main_')]
            if not main_data.empty:
                main_1min = main_data[main_data['data_type'] == 'main_1min']
                main_5min = main_data[main_data['data_type'] == 'main_5min']

                result['main_contract'] = {
                    'code': main_1min['Code'].iloc[0] if not main_1min.empty else (
                        main_5min['Code'].iloc[0] if not main_5min.empty else None
                    ),
                    'multiplier_1min': main_1min['multiplier'].iloc[0] if not main_1min.empty else 1.0,
                    'multiplier_5min': main_5min['multiplier'].iloc[0] if not main_5min.empty else 1.0,
                }

            # 次主力合约（data_type 以 'sub_' 开头）
            sub_data = df_date[df_date['data_type'].str.startswith('sub_')]
            if not sub_data.empty:
                sub_1min = sub_data[sub_data['data_type'] == 'sub_1min']
                sub_5min = sub_data[sub_data['data_type'] == 'sub_5min']

                result['sub_contract'] = {
                    'code': sub_1min['Code'].iloc[0] if not sub_1min.empty else (
                        sub_5min['Code'].iloc[0] if not sub_5min.empty else None
                    ),
                    'multiplier_1min': sub_1min['multiplier'].iloc[0] if not sub_1min.empty else 1.0,
                    'multiplier_5min': sub_5min['multiplier'].iloc[0] if not sub_5min.empty else 1.0,
                }

            return result

        except Exception as e:
            print(f"[DataAPI] 加载连续乘数失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def close(self):
        """关闭连接"""
        if self._ddb_session:
            try:
                self._ddb_session.close()
            except Exception:
                pass
            self._ddb_session = None
