"""
BacktestRunner - 引擎封装
将 Nautilus BacktestEngine 包装为 bomber 风格的使用接口。

使用方式（与生产类似）：
    runner = BacktestRunner(config={...})
    runner.add_instrument(instrument)
    runner.load_bars('IF2406', 'M1', filepath)
    runner.load_ticks('IF2406', filepath)

    stg = MyStrategy()
    stg.initialize('MyStrategy', env)
    stg.subInstrument('IF2406')
    stg.subBar('IF2406', 'M1')

    result = runner.run(stg)
"""
import re
import pandas as pd
from decimal import Decimal
from typing import List

from bomber.backtest.engine import BacktestEngine
from bomber.config import BacktestEngineConfig, LoggingConfig
from bomber.model import TraderId
from bomber.model.currencies import CNY
from bomber.model.data import Bar, BarType, BarSpecification, BarAggregation
from bomber.model.data import QuoteTick, TradeTick
from bomber.model.enums import AccountType, OmsType
from bomber.model.identifiers import InstrumentId, Symbol, Venue
from bomber.model.objects import Money, Price, Quantity
from bomber.model.instruments.futures_contract import FuturesContract
from bomber.persistence.wranglers import BarDataWrangler

from .bridge import BomberBridge
from .strategy import IPyStrategy
from .data_loader import FileDataLoader
from .data_api import DataAPI
from .enums import PERIOD_MAP


class BacktestRunner:
    """
    回测运行器 - 封装 Nautilus BacktestEngine。

    提供 bomber 风格的使用接口：
      - add_instrument: 添加合约
      - load_bars: 加载K线数据
      - load_ticks: 加载Tick数据
      - run: 运行回测
    """

    def __init__(self, config: dict = None, config_file: str = None):
        """
        初始化回测运行器

        参数:
            config: 配置字典（优先级高）
            config_file: 配置文件路径（JSON 格式，优先级低）

        合并规则: config_file 作为 baseline，config 中的值覆盖 config_file。
        """
        # 如果提供了配置文件，先加载作为 baseline
        file_config = {}
        if config_file:
            from .config_loader import load_config
            file_config = load_config(config_file)

        # 合并：文件配置作为基础，代码配置覆盖
        merged = {}
        merged.update(file_config)
        if config:
            merged.update(config)

        # 解析配置（支持新的嵌套格式和旧的扁平格式）
        self._parse_config(merged)

    def _parse_config(self, config: dict):
        """
        解析配置（支持新的嵌套格式和旧的扁平格式）
        """
        # 新的嵌套格式（JSON）
        if "venue" in config and isinstance(config["venue"], dict):
            # 新格式
            venue_config = config["venue"]
            self._venue_str = venue_config.get("name", "CFFEX")
            self._oms_type = OmsType[venue_config.get("oms_type", "NETTING")]
        else:
            # 旧格式（扁平）
            self._venue_str = config.get("venue", "CFFEX")
            self._oms_type = OmsType[config.get("oms_type", "NETTING")]

        self._venue = Venue(self._venue_str)
        self._data_dir = config.get("data_dir", "")

        # 回测时间范围
        self._period_start = config.get("period_start")
        self._period_end = config.get("period_end")

        # 账户配置
        if "account" in config and isinstance(config["account"], dict):
            account_config = config["account"]
            self._starting_balance = float(account_config.get("initial_capital", 10_000_000))
            self._currency = account_config.get("currency", "CNY")
        else:
            self._starting_balance = float(config.get("starting_balance", 10_000_000))
            self._currency = config.get("base_currency", "CNY")

        self._log_level = config.get("log_level", "INFO")

        # 撮合模型配置
        self._fill_model_config = config.get("fill_model")
        self._latency_model_config = config.get("latency_model")
        self._fee_model_config = config.get("fee_model")

        # 撮合执行配置
        if "matching" in config and isinstance(config["matching"], dict):
            matching_config = config["matching"]
            self._bar_execution = matching_config.get("bar_execution", True)
            self._trade_execution = matching_config.get("trade_execution", True)
            self._bar_adaptive_high_low_ordering = matching_config.get("bar_adaptive_high_low_ordering", False)
        else:
            self._bar_execution = config.get("bar_execution", True)
            self._trade_execution = config.get("trade_execution", True)
            self._bar_adaptive_high_low_ordering = config.get("bar_adaptive_high_low_ordering", False)

        # 成本配置
        self._cost_config = config.get("cost", {})

        # 结算配置
        self._settlement_prices = config.get("settlement_prices")

        # 创建 Nautilus 引擎
        engine_config = BacktestEngineConfig(
            trader_id=TraderId(config.get("trader_id", "BACKTESTER-001")),
            logging=LoggingConfig(log_level=self._log_level),
        )
        self._engine = BacktestEngine(config=engine_config)

        # 添加交易所（带撮合模型配置）
        venue_kwargs = {
            "venue": self._venue,
            "oms_type": self._oms_type,
            "account_type": AccountType.MARGIN,
            "starting_balances": [Money(self._starting_balance, CNY)],
            "base_currency": CNY,
            "default_leverage": Decimal(1),
            "bar_execution": self._bar_execution,
            "trade_execution": self._trade_execution,
            "bar_adaptive_high_low_ordering": self._bar_adaptive_high_low_ordering,
        }

        # 添加撮合模型
        if self._fill_model_config:
            from bomber.backtest.config import ImportableFillModelConfig, FillModelFactory
            importable_config = ImportableFillModelConfig(
                fill_model_path=self._fill_model_config["path"],
                config_path=self._fill_model_config["config_path"],
                config=self._fill_model_config["config"],
            )
            venue_kwargs["fill_model"] = FillModelFactory.create(importable_config)

        # 添加延迟模型
        if self._latency_model_config:
            from bomber.backtest.config import ImportableLatencyModelConfig, LatencyModelFactory
            importable_config = ImportableLatencyModelConfig(
                latency_model_path=self._latency_model_config["path"],
                config_path=self._latency_model_config["config_path"],
                config=self._latency_model_config["config"],
            )
            venue_kwargs["latency_model"] = LatencyModelFactory.create(importable_config)

        # 添加费用模型
        if self._fee_model_config:
            from bomber.backtest.config import ImportableFeeModelConfig, FeeModelFactory
            importable_config = ImportableFeeModelConfig(
                fee_model_path=self._fee_model_config["path"],
                config_path=self._fee_model_config["config_path"],
                config=self._fee_model_config["config"],
            )
            venue_kwargs["fee_model"] = FeeModelFactory.create(importable_config)

        # 添加结算价格
        if self._settlement_prices:
            from bomber.model.identifiers import InstrumentId
            # 将字符串键转换为 InstrumentId
            settlement_prices = {}
            for inst_id_str, price in self._settlement_prices.items():
                if "." in inst_id_str:
                    settlement_prices[InstrumentId.from_str(inst_id_str)] = price
                else:
                    settlement_prices[InstrumentId(Symbol(inst_id_str), self._venue)] = price
            venue_kwargs["settlement_prices"] = settlement_prices

        self._engine.add_venue(**venue_kwargs)

        # 数据加载器
        self._data_loader = FileDataLoader(self._data_dir, self._venue_str)
        self._data_api = DataAPI(self._data_dir)

        # 加载的数据
        self._all_data = []
        self._instruments = {}
        self._streaming = config.get("streaming", False)  # 流式加载（低内存）

    def add_instrument(self, instrument):
        """添加合约定义"""
        self._engine.add_instrument(instrument)
        self._instruments[str(instrument.id.symbol)] = instrument

    def add_futures_contract(self, symbol: str, expiry_year: int,
                               expiry_month: int, multiplier: float = 200.0,
                               price_precision: int = 2,
                               tick_size: float = 0.2):
        """
        添加股指期货合约。

        参数:
            symbol:          合约代码 (如 "IC2406")
            expiry_year:     到期年份
            expiry_month:    到期月份
            multiplier:      合约乘数
            price_precision: 价格精度
            tick_size:       最小变动价位
        """
        from bomber.model.enums import AssetClass
        import calendar
        from datetime import datetime, timezone

        # 计算到期日（每月第三个周五，UTC 时间）
        cal = calendar.monthcalendar(expiry_year, expiry_month)
        fridays = [week[4] for week in cal if week[4] != 0]
        expiry_day = fridays[2] if len(fridays) >= 3 else fridays[-1]
        expiry_dt = datetime(expiry_year, expiry_month, expiry_day, 15, 0, tzinfo=timezone.utc)
        expiration_ns = int(expiry_dt.timestamp() * 1e9)
        # 激活日：到期前 12 个月（覆盖合约上市到到期的完整生命周期）
        act_year = expiry_year - 1 if expiry_month <= 6 else expiry_year
        act_month = expiry_month + 6 if expiry_month <= 6 else expiry_month - 6
        # act_month 最大为 12（expiry_month=6 时），无需 >12 检查
        activation_ns = int(datetime(act_year, act_month, 1, tzinfo=timezone.utc).timestamp() * 1e9)

        # 提取品种代码：去掉末尾数字（如 "IC2406" → "IC", "T2409" → "T"）
        underlying = re.match(r'^([A-Za-z]+)', symbol)
        underlying = underlying.group(1) if underlying else symbol

        inst = FuturesContract(
            instrument_id=InstrumentId(Symbol(symbol), self._venue),
            raw_symbol=Symbol(symbol),
            asset_class=AssetClass.INDEX,
            currency=CNY,
            price_precision=price_precision,
            price_increment=Price(tick_size, price_precision),
            multiplier=Quantity(multiplier, 0),
            lot_size=Quantity(1, 0),
            underlying=underlying,
            activation_ns=activation_ns,
            expiration_ns=expiration_ns,
            ts_event=0,
            ts_init=0,
        )
        self.add_instrument(inst)
        return inst

    def load_bars(self, instrument_id: str, period: str = "M1",
                  filepath: str = None) -> list:
        """加载K线数据。streaming 模式只注册不加载（由 generator 直读 parquet）"""
        if self._streaming:
            return []
        instrument = self._instruments.get(instrument_id)
        # 批次模式扩展 period_end 1 天（与 _monthly_chunks 保持一致，半开区间包含最后一天）
        import pandas as pd
        pe = self._period_end
        if pe:
            pe = (pd.Timestamp(pe) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        bars = self._data_loader.load_bars(
            instrument_id, period, filepath, instrument=instrument,
            period_start=self._period_start, period_end=pe)
        self._all_data.extend(bars)
        return bars

    def load_ticks(self, instrument_id: str, filepath: str = None) -> tuple:
        """加载Tick数据"""
        quotes, trades = self._data_loader.load_ticks(instrument_id, filepath)
        self._all_data.extend(quotes)
        self._all_data.extend(trades)
        return quotes, trades

    def add_data(self, data_list: list):
        """添加数据"""
        self._all_data.extend(data_list)

    def get_data_api(self) -> DataAPI:
        """获取数据API（供策略启动时加载数据）"""
        return self._data_api

    def run(self, strategy: IPyStrategy) -> dict:
        """
        运行回测。

        参数:
            strategy: IPyStrategy 实例

        返回:
            dict: 回测结果摘要
        """
        # 1. 将数据加入引擎
        if self._streaming:
            gen = _parquet_bar_generator(
                self._data_dir, self._instruments,
                self._period_start, self._period_end)
            self._engine.add_data_iterator("bars", gen)
            self._all_data.clear()  # 不再需要
        else:
            self._all_data.sort(key=lambda d: d.ts_event)
            self._engine.add_data(self._all_data)

        # 2. 创建桥接器（Nautilus Strategy 包装用户的 IPyStrategy）
        bridge = BomberBridge(
            user_strategy=strategy,
            data_api=self._data_api,
            data_dir=self._data_dir,
        )
        self._engine.add_strategy(bridge)

        # 3. 运行 + 提取 + 清理（finally 确保异常时也能 dispose）
        try:
            self._engine.run()
            result = self._extract_result()
            try:
                self._cached_orders = self._engine.cache.orders()
            except Exception:
                self._cached_orders = []

            # 在 dispose 之前获取权益曲线和交易记录
            try:
                self._cached_equity_curve = self.get_equity_curve()
                if self._cached_equity_curve is not None and not self._cached_equity_curve.empty:
                    import pandas as pd
                    eq = self._cached_equity_curve
                    tz = eq.index.tz
                    # 插入起点锚点，确保 resample 对齐（batch/stream 原始起点可能不同）
                    if self._period_start:
                        ps = pd.Timestamp(self._period_start)
                        if tz:
                            ps = ps.tz_localize('Asia/Shanghai').tz_convert(tz)
                        if ps not in eq.index:
                            eq.loc[ps] = float(eq.iloc[0])
                            eq = eq.sort_index()
                    # 重采样为日频（统一 batch/stream 的 snapshot 频率差异）
                    if isinstance(eq.index, pd.DatetimeIndex):
                        eq = eq.resample('D').last().ffill().dropna()
                    # 截断到 period_start/period_end
                    if self._period_start:
                        ts = pd.Timestamp(self._period_start)
                        if tz:
                            ts = ts.tz_localize('Asia/Shanghai').tz_convert(tz)
                        eq = eq[eq.index >= ts]
                    if self._period_end and not eq.empty:
                        te = pd.Timestamp(self._period_end) + pd.Timedelta(days=1)
                        if tz:
                            te = te.tz_localize('Asia/Shanghai').tz_convert(tz)
                        eq = eq[eq.index < te]
                    self._cached_equity_curve = eq

                    # 使用权益曲线终点更新 ending_equity（修复 stream 模式下 cache.accounts() 不准确的问题）
                    if not eq.empty:
                        ending_equity = float(eq.iloc[-1])
                        result['ending_equity'] = ending_equity
                        result['total_pnl'] = ending_equity - result['starting_balance']
            except Exception:
                self._cached_equity_curve = None

            try:
                self._cached_trades = self.get_trades()
            except Exception:
                self._cached_trades = []

        finally:
            self._engine.dispose()

        return result

    def get_equity_curve(self) -> pd.Series:
        """
        获取权益曲线

        返回:
            pd.Series: 时间序列，索引为时间，值为权益
        """
        try:
            import pandas as pd
            reports = self._engine.trader.generate_account_report(self._venue)
            if not reports.empty:
                # 尝试多个可能的列名
                for col in ['total', 'equity', 'balance', 'net_value']:
                    if col in reports.columns:
                        series = reports[col]
                        # 转换为数值类型
                        return pd.to_numeric(series, errors='coerce').dropna()
                # 如果没有找到，返回第一列数值列
                for col in reports.columns:
                    if reports[col].dtype in ['float64', 'int64']:
                        return reports[col]
                    # 尝试转换为数值
                    try:
                        series = pd.to_numeric(reports[col], errors='coerce')
                        if not series.isna().all():
                            return series.dropna()
                    except:
                        continue
        except Exception as e:
            print(f"[BacktestRunner] 获取权益曲线失败: {e}")
        return pd.Series(dtype=float)

    def get_trades(self) -> list:
        """
        获取交易记录

        返回:
            list: 交易记录列表，每条包含时间、合约、方向、数量、价格、盈亏
        """
        trades = []
        try:
            fills = getattr(self, '_cached_orders', [])
            if not fills:
                return trades

            for order in fills:
                try:
                    f_qty = float(order.filled_qty.as_double()) if hasattr(order.filled_qty, 'as_double') else float(order.filled_qty)
                except Exception:
                    continue
                if f_qty <= 0:
                    continue
                ts = getattr(order, 'ts_last', None) or getattr(order, 'ts_accepted', 0)
                trade = {
                    'timestamp': int(ts),
                    'instrument': str(order.instrument_id),
                    'side': 'BUY' if order.is_buy else 'SELL',
                    'qty': f_qty,
                    'price': float(order.avg_px.as_double()) if hasattr(order.avg_px, 'as_double') else float(order.avg_px),
                    'pnl': 0.0,
                }
                trades.append(trade)
        except Exception as e:
            print(f"[BacktestRunner] 获取交易记录失败: {e}")
            import traceback
            traceback.print_exc()
        return trades

    def _extract_result(self) -> dict:
        """从引擎提取回测结果"""
        cache = self._engine.cache

        try:
            accounts = cache.accounts()
            total_equity = 0.0
            if accounts:
                # 汇总所有账户的余额
                for acct in accounts:
                    for currency, money in acct.balances_total().items():
                        total_equity += float(money.as_double())
            if total_equity == 0:
                total_equity = self._starting_balance
        except Exception:
            total_equity = self._starting_balance

        try:
            all_orders = cache.orders()
            open_orders = cache.orders_open()
            closed_orders = cache.orders_closed()
            open_positions = cache.positions_open()
            closed_positions = cache.positions_closed()
        except Exception:
            all_orders = []
            open_orders = []
            closed_orders = []
            open_positions = []
            closed_positions = []

        return {
            "starting_balance": self._starting_balance,
            "ending_equity": round(total_equity, 2),
            "total_pnl": round(total_equity - self._starting_balance, 2),
            "total_orders": len(all_orders),
            "orders_open": len(open_orders),
            "orders_closed": len(closed_orders),
            "positions_open": len(open_positions),
            "positions_closed": len(closed_positions),
            "total_data_events": len(self._all_data),
        }

    def export_fills(self, filepath: str) -> str:
        """导出成交记录为 CSV。返回文件路径，无成交时返回空字符串。"""
        import pandas as pd

        # 优先使用缓存的订单列表
        fills = getattr(self, '_cached_orders', None)
        if fills is None:
            try:
                fills = self._engine.cache.orders()
            except Exception:
                fills = []
        # 确保是 list（cache.orders 可能返回 Cython generator）
        try:
            fills = list(fills)
        except Exception:
            fills = []

        print(f"[BacktestRunner] Exporting {len(fills)} orders...")
        if not fills:
            return ""

        rows = []
        errors = 0
        for o in fills:
            try:
                ts = getattr(o, 'ts_accepted', getattr(o, 'ts_init', 0))
                qty = o.filled_qty
                f_qty = float(qty.as_double()) if hasattr(qty, 'as_double') else float(qty)
                if f_qty <= 0:
                    continue
                price = o.avg_px
                f_price = float(price.as_double()) if hasattr(price, 'as_double') else float(price)
                inst_str = str(o.instrument_id)
                side_str = "BUY" if "BUY" in str(o.side) else "SELL"
                rows.append({
                    "timestamp": pd.Timestamp(int(ts), unit='ns').strftime('%Y-%m-%d %H:%M:%S'),
                    "instrument_id": inst_str,
                    "side": side_str,
                    "qty": f_qty,
                    "price": f_price,
                })
            except Exception:
                errors += 1
                continue

        if errors:
            print(f"[BacktestRunner] {errors} orders skipped due to errors")

        if not rows:
            return ""

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"[BacktestRunner] Exported {len(df)} fills → {filepath}")
        return filepath


def _parquet_bar_generator(data_dir: str, instruments: dict,
                            period_start: str = None, period_end: str = None):
    """按月分片，每月加载全部合约、排序后逐批 yield（保序 + 低内存）"""
    from .data_loader import FileDataLoader

    loader = FileDataLoader(data_dir)
    inst_ids = list(instruments.keys())
    chunks = _monthly_chunks(period_start, period_end)

    for chunk_start, chunk_end in chunks:
        month_bars = []
        for inst_id in inst_ids:
            bars = loader.load_bars(
                inst_id, "M1", instrument=instruments.get(inst_id),
                period_start=chunk_start, period_end=chunk_end)
            if bars:
                month_bars.extend(bars)

        if not month_bars:
            continue
        month_bars.sort(key=lambda b: b.ts_event)
        for i in range(0, len(month_bars), 5000):
            yield month_bars[i:i + 5000]
        del month_bars


def _monthly_chunks(start: str, end: str):
    """将日期范围按月切分"""
    from datetime import datetime, timedelta
    if not start or not end:
        return [(start, end)]
    d = datetime.strptime(start[:10], "%Y-%m-%d")
    e = datetime.strptime(end[:10], "%Y-%m-%d")
    chunks = []
    while d < e:
        nxt = (d.replace(day=1) + timedelta(days=32)).replace(day=1)
        if nxt >= e:
            nxt = e + timedelta(days=1)  # 半开区间，包含最后一天
        chunks.append((d.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        d = nxt
    return chunks
