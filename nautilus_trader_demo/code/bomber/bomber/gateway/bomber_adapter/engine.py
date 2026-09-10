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
import os
import copy
import pandas as pd
from decimal import Decimal
from typing import List

from bomber.backtest.engine import BacktestEngine
from bomber.config import BacktestEngineConfig, LoggingConfig
from bomber.model import TraderId
from bomber.model.currencies import CNY
from bomber.model.data import Bar, BarType, BarSpecification, BarAggregation
from bomber.model.data import QuoteTick, TradeTick
from bomber.model.enums import AccountType, OmsType, OrderSide
from bomber.model.identifiers import InstrumentId, Symbol, Venue
from bomber.model.objects import Money, Price, Quantity
from bomber.model.instruments.futures_contract import FuturesContract
from bomber.persistence.wranglers import BarDataWrangler

from bomber_adapter.bridge import BomberBridge
from bomber_adapter.strategy import IPyStrategy
from bomber_adapter.data_loader import FileDataLoader
from bomber_adapter.data_api import DataAPI
from bomber_adapter.enums import PERIOD_MAP
from bomber_adapter.instrument_info import is_option


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge configuration mappings without dropping sibling keys."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _order_is_buy(order) -> bool:
    """Version-stable order direction lookup."""
    side = getattr(order, "side", None)
    if side is not None:
        return side == OrderSide.BUY
    side_string = getattr(order, "side_string", None)
    if callable(side_string):
        return side_string().upper() == "BUY"
    value = getattr(order, "is_buy", False)
    return bool(value() if callable(value) else value)


def _order_fill_timestamp_ns(order) -> int:
    """Use fill/close time before accepted/submitted time."""
    for name in ("ts_closed", "ts_last", "ts_accepted", "ts_init"):
        value = int(getattr(order, name, 0) or 0)
        if value > 0:
            return value
    return 0


def _parquet_for_inst(inst: str, period: str = "M1") -> str:
    """
    根据合约代码和周期确定 parquet 文件名。

    统一走 data_loader.resolve_pq_name（唯一事实来源）。
    """
    from bomber_adapter.data_loader import resolve_pq_name
    return resolve_pq_name(inst, period)


def _compute_commission(inst: str, qty: float, price: float, multiplier: float,
                        futures_bps: float, option_per_contract: float) -> float:
    """
    根据合约类型计算手续费：
      - 期货：成交金额 × futures_bps / 10000
      - 期权：qty × option_per_contract（每张合约固定费用）
    """
    if is_option(inst):
        return qty * option_per_contract
    else:
        notional = qty * price * multiplier
        return notional * futures_bps / 10000.0


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
            from bomber_adapter.config_loader import load_config
            file_config = load_config(config_file)

        # 合并：文件配置作为基础，代码配置覆盖
        merged = _deep_merge(file_config, config or {})

        # 解析配置（支持新的嵌套格式和旧的扁平格式）
        self._parse_config(merged)

    def _parse_config(self, config: dict):
        """
        解析配置（支持新的嵌套格式和旧的扁平格式）
        """
        self._config = config  # 保存完整配置供后续使用

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
        self._data_dir = (
            config.get("data_dir")
            or os.environ.get("BOMBER_DATA_DIR")
            or os.environ.get("BT_DATA_DIR")
            or "./data"
        )

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

        execution_config = config.get("execution", {})
        if not isinstance(execution_config, dict):
            raise TypeError("execution must be a mapping")
        self._execution_delay_bars = int(execution_config.get("delay_bars", 0))
        if self._execution_delay_bars < 0:
            raise ValueError("execution.delay_bars must be non-negative")
        data_config = config.get("data", {})
        if not isinstance(data_config, dict):
            raise TypeError("data must be a mapping")
        self._bar_timestamp_mode = data_config.get(
            "bar_timestamp",
            config.get("bar_timestamp", "end"),
        )
        if self._bar_timestamp_mode not in {"start", "end"}:
            raise ValueError("data.bar_timestamp must be 'start' or 'end'")
        self._fee_model_config = config.get("fee_model")
        reporting_config = config.get("reporting", {})
        if not isinstance(reporting_config, dict):
            raise TypeError("reporting must be a mapping")
        self._equity_source = reporting_config.get("equity_source", "nautilus")
        if self._equity_source not in {"nautilus", "custom"}:
            raise ValueError("reporting.equity_source must be 'nautilus' or 'custom'")
        self._strict_data_validation = bool(
            reporting_config.get("strict_data_validation", True),
        )
        # 是否在权益曲线中扣减手续费（默认 false，对齐第三方费前 NAV 口径）
        self._include_commission = bool(
            reporting_config.get("include_commission", False),
        )

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
        # 解析分品种手续费：
        #   - futures_bps: 期货按成交额万分之几（默认取 commission_bps，向后兼容）
        #   - option_per_contract: 期权每张合约固定费用（CNY）
        self._futures_bps = float(
            self._cost_config.get("futures_bps",
                                   self._cost_config.get("commission_bps", 0.0))
        )
        self._option_per_contract = float(
            self._cost_config.get("option_per_contract", 0.0)
        )

        # 结算配置（可从文件自动加载）
        self._settlement_prices = config.get("settlement_prices")
        if not self._settlement_prices and config.get("settlement_eod", False):
            self._settlement_prices = _load_option_settlement(self._data_dir)

        # 创建 Nautilus 引擎
        engine_config = BacktestEngineConfig(
            trader_id=TraderId(config.get("trader_id", "BACKTESTER-001")),
            logging=LoggingConfig(log_level=self._log_level),
        )
        self._engine = BacktestEngine(config=engine_config)

        # 添加交易所（带撮合模型配置）
        venue_config = self._config.get("venue", {})
        if not isinstance(venue_config, dict):
            venue_config = {}
        venue_kwargs = {
            "venue": self._venue,
            "oms_type": self._oms_type,
            "account_type": AccountType.MARGIN,
            "starting_balances": [Money(self._starting_balance, CNY)],
            "base_currency": CNY,
            "default_leverage": Decimal(venue_config.get("default_leverage", 1)),
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

        # LatencyModel 只模拟真实时间延迟；next-bar 由 execution.delay_bars 控制。
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
        self._data_loader = FileDataLoader(
            self._data_dir,
            self._venue_str,
            bar_timestamp=self._bar_timestamp_mode,
        )
        self._data_api = DataAPI(self._data_dir)

        # 加载的数据
        self._all_data = []
        self._instruments = {}
        self._instrument_multipliers = {}
        self._option_expiry_dates = {}
        self._futures_expiry_dates = {}  # 期货到期日
        self._bar_periods = {}
        self._streaming = config.get("streaming", False)  # 流式加载（低内存）

        metadata_dir = os.environ.get("BOMBER_METADATA_DIR") or os.path.join(self._data_dir, 'metadata')

        # 加载预计算的到期日数据（期货+期权）
        try:
            expiry_path = os.path.join(metadata_dir, 'expiry_dates.parquet')
            if os.path.exists(expiry_path):
                _expiry = pd.read_parquet(expiry_path)
                # 分离期货和期权
                futures_expiry = _expiry[_expiry['instrument_type'] == 'FUTURE']
                options_expiry = _expiry[_expiry['instrument_type'] == 'OPTION']

                # 期货到期日
                self._futures_expiry_dates = dict(
                    zip(futures_expiry['code'], futures_expiry['expiry_date'])
                )

                # 期权到期日（合并到 _option_expiry_dates）
                for _, row in options_expiry.iterrows():
                    self._option_expiry_dates[row['code']] = row['expiry_date']

                print(f"[BacktestRunner] 加载到期日数据: "
                      f"期货 {len(self._futures_expiry_dates)} 个, "
                      f"期权 {len(self._option_expiry_dates)} 个")
            else:
                print(f"[BacktestRunner] 未找到到期日数据文件: {expiry_path}")
                print(f"[BacktestRunner] 请运行 tools/generate_expiry_dates.py 生成")
        except Exception as exc:
            print(f"[BacktestRunner] 无法加载到期日数据: {exc}")

        # 期权真实最后交易日 {code: "YYYY-MM-DD"}，来自 opt_eod.parquet。
        # 优先于"第三周五"算法（节假日顺延时第三周五≠最后交易日，如 2024-02 春节）。
        self._opt_last_trade_day = {}
        try:
            settlement_path = os.path.join(metadata_dir, 'opt_eod.parquet')
            if os.path.exists(settlement_path):
                _eod = pd.read_parquet(
                    settlement_path,
                    columns=['tradeID', 'lastTradingDate'])
                _eod = _eod.dropna()
                _eod['ltd'] = pd.to_datetime(
                    _eod['lastTradingDate']).dt.strftime('%Y-%m-%d')
                self._opt_last_trade_day = _eod.groupby('tradeID')['ltd'].max().to_dict()
        except Exception as exc:
            print(f"[BacktestRunner] 无法加载期权最后交易日(opt_eod): {exc}")

        # 交易日历（用于校正到期日：第三周五遇到假期时顺延到下一个交易日）
        self._trading_days = set()
        try:
            cal_path = os.path.join(metadata_dir, 'calendar.csv')
            if os.path.exists(cal_path):
                _cal = pd.read_csv(cal_path)
                _cal['date'] = pd.to_datetime(_cal['date'])
                self._trading_days = set(
                    _cal[_cal['is_trading_day']]['date'].dt.date
                )
        except Exception as exc:
            print(f"[BacktestRunner] 无法加载交易日历(calendar.csv): {exc}")

    def add_instrument(self, instrument):
        """添加合约定义"""
        self._engine.add_instrument(instrument)
        symbol = str(instrument.id.symbol)
        self._instruments[symbol] = instrument
        multiplier = getattr(instrument, "multiplier", None)
        if multiplier is not None:
            self._instrument_multipliers[symbol] = (
                float(multiplier.as_double())
                if hasattr(multiplier, "as_double")
                else float(multiplier)
            )

    def _instrument_multiplier(self, symbol: str) -> float:
        """Return the registered multiplier; never infer it from the symbol."""
        code = symbol.split(".")[0]
        try:
            return float(self._instrument_multipliers[code])
        except KeyError as exc:
            raise KeyError(f"No registered multiplier for {code}") from exc

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

        # 不让 Nautilus 自动到期结算（持仓文件控制退出时机）
        activation_ns = 0
        expiration_ns = 2**63 - 1

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

    def add_option_contract(self, symbol: str, expiry_year: int, expiry_month: int,
                             expiry_day: int = 0, strike: float = 0.0,
                             option_type: str = "CALL", multiplier: float = 100.0,
                             price_precision: int = 4, tick_size: float = 0.0001):
        """添加 ETF/股指期权合约"""
        from bomber.model.instruments import OptionContract
        from bomber.model.enums import OptionKind, AssetClass
        import calendar
        from datetime import datetime, timezone

        if expiry_day <= 0:
            # 优先使用 opt_eod 的真实最后交易日（节假日顺延时第三周五≠最后交易日）
            real_day = (self._opt_last_trade_day or {}).get(symbol)
            if real_day:
                year, month, day = (int(x) for x in real_day.split('-'))
                expiry_day = day
                expiry_year, expiry_month = year, month
            else:
                cal = calendar.monthcalendar(expiry_year, expiry_month)
                fridays = [week[4] for week in cal if week[4] != 0]
                expiry_day = fridays[2] if len(fridays) >= 3 else fridays[-1]  # 第三个周五
                # 交易日历校正：第三周五遇到假期时顺延到下一个交易日
                if self._trading_days:
                    from datetime import date, timedelta
                    d = date(expiry_year, expiry_month, expiry_day)
                    max_days = 10  # 最多顺延 10 天（覆盖长假）
                    for _ in range(max_days):
                        if d in self._trading_days:
                            break
                        d += timedelta(days=1)
                    expiry_year, expiry_month, expiry_day = d.year, d.month, d.day

        option_kind = OptionKind.CALL if option_type.upper() in ("CALL", "C", "认购") else OptionKind.PUT
        # MO2301→IM2301, IO2301→IF2301, HO2301→IH2301
        root = symbol.split("-")[0] if "-" in symbol else symbol[:2]
        root_clean = re.match(r'^([A-Za-z]+)', root).group(1) if root else root
        ul_map = {"MO": "IM", "IO": "IF", "HO": "IH"}
        ul_product = ul_map.get(root_clean, root_clean)
        underlying = ul_product + root[-4:]

        # 期权到期结算由 _add_option_settlement 用本地的 opt_eod.parquet 结算价处理，
        # 不让 Nautilus 引擎自动结算（避免 No underlying price 错误）。
        # expiration_ns 设为永不自然到期。
        activation_ns = 0
        expiration_ns = 2**63 - 1

        inst = OptionContract(
            instrument_id=InstrumentId(Symbol(symbol), self._venue),
            raw_symbol=Symbol(symbol),
            asset_class=AssetClass.INDEX,
            currency=CNY,
            price_precision=price_precision,
            price_increment=Price(tick_size, price_precision),
            multiplier=Quantity(multiplier, 0),
            lot_size=Quantity(1, 0),
            underlying=underlying,
            option_kind=option_kind,
            strike_price=Price(strike, price_precision),
            activation_ns=activation_ns,
            expiration_ns=expiration_ns,
            ts_event=0,
            ts_init=0,
        )
        self.add_instrument(inst)
        self._option_expiry_dates[symbol] = (
            f"{expiry_year:04d}-{expiry_month:02d}-{expiry_day:02d}"
        )
        return inst

    def load_bars(self, instrument_id: str, period: str = "M1",
                  filepath: str = None) -> list:
        """加载K线数据。streaming 模式只注册不加载（由 generator 直读 parquet）"""
        if self._streaming:
            normalized_period = period.upper()
            existing = self._bar_periods.get(instrument_id)
            if existing is not None and existing != normalized_period:
                raise ValueError(
                    f"Streaming supports one period per instrument: "
                    f"{instrument_id} has {existing} and {normalized_period}",
                )
            self._bar_periods[instrument_id] = normalized_period
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
            # 优化：如果数据已经加载到_all_data中，直接使用，不要重新加载
            if self._all_data:
                self._all_data.sort(key=lambda d: d.ts_event)
                self._engine.add_data(self._all_data)
            else:
                # 如果没有预加载数据，才使用streaming generator
                gen = _parquet_bar_generator(
                    self._data_dir, self._instruments,
                    self._bar_periods,
                    self._period_start, self._period_end,
                    self._venue_str,
                    self._bar_timestamp_mode,
                    data_loader=self._data_loader,  # 传递已有的 data_loader 以使用缓存
                )
                self._engine.add_data_iterator("bars", gen)
        else:
            self._all_data.sort(key=lambda d: d.ts_event)
            self._engine.add_data(self._all_data)

        # 2. 创建桥接器（Nautilus Strategy 包装用户的 IPyStrategy）
        bridge = BomberBridge(
            user_strategy=strategy,
            data_api=self._data_api,
            data_dir=self._data_dir,
            execution_delay_bars=self._execution_delay_bars,
            venue=self._venue_str,
            config=self._config,
            option_expiry_dates=self._option_expiry_dates,
            futures_expiry_dates=self._futures_expiry_dates,
            bar_extra_fields=self._data_loader.get_bar_extra_fields(),
        )
        self._bridge = bridge  # 保存引用，用于导出持仓快照
        self._engine.add_strategy(bridge)

        # 3. 运行 + 提取 + 清理（finally 确保异常时也能 dispose）
        try:
            self._engine.run()
            result = self._extract_result()
            result["nautilus_ending_equity"] = result["ending_equity"]
            result["nautilus_total_pnl"] = result["total_pnl"]
            try:
                self._cached_orders = self._engine.cache.orders()
            except Exception:
                self._cached_orders = []

            # 在 dispose 之前获取权益曲线和交易记录
            try:
                self._cached_equity_curve = self._build_equity_from_fills(self._starting_balance)
                eq = self._cached_equity_curve
                if eq is not None and not eq.empty:
                    custom_ending = float(eq.iloc[-1])
                    result["custom_ending_equity"] = custom_ending
                    result["custom_total_pnl"] = custom_ending - result["starting_balance"]
                    if self._equity_source == "custom":
                        result["ending_equity"] = custom_ending
                        result["total_pnl"] = custom_ending - result["starting_balance"]
                result["equity_source"] = self._equity_source
            except Exception as e:
                import traceback
                print(f"[BacktestRunner] 权益曲线构建失败: {e}")
                traceback.print_exc()
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
                ts = _order_fill_timestamp_ns(order)
                trade = {
                    'timestamp': int(ts),
                    'instrument': str(order.instrument_id),
                    'side': 'BUY' if _order_is_buy(order) else 'SELL',
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
        from datetime import timezone, timedelta

        # 定义北京时间时区（UTC+8）
        CST = timezone(timedelta(hours=8))

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
                ts = _order_fill_timestamp_ns(o)
                qty = o.filled_qty
                f_qty = float(qty.as_double()) if hasattr(qty, 'as_double') else float(qty)
                if f_qty <= 0:
                    continue
                price = o.avg_px
                f_price = float(price.as_double()) if hasattr(price, 'as_double') else float(price)
                inst_str = str(o.instrument_id)
                inst_code = inst_str.split('.')[0]
                side_str = "BUY" if _order_is_buy(o) else "SELL"
                commission = 0.0
                if self._futures_bps > 0 or self._option_per_contract > 0:
                    mul = self._instrument_multiplier(inst_code)
                    commission = _compute_commission(
                        inst_code, f_qty, f_price, mul,
                        self._futures_bps, self._option_per_contract,
                    )
                else:
                    commissions = getattr(o, "commissions", None)
                    if callable(commissions):
                        for money in commissions():
                            commission += (
                                float(money.as_double())
                                if hasattr(money, "as_double")
                                else float(money)
                            )
                # 将 UTC 时间戳转换为北京时间
                dt_utc = pd.Timestamp(int(ts), unit='ns', tz='UTC')
                dt_cst = dt_utc.tz_convert(CST)
                rows.append({
                    "timestamp": dt_cst.strftime('%Y-%m-%d %H:%M:%S'),
                    "instrument_id": inst_str,
                    "order_id": str(o.client_order_id),
                    "side": side_str,
                    "qty": f_qty,
                    "price": f_price,
                    "commission": commission,
                })
            except Exception:
                errors += 1
                continue

        if errors:
            print(f"[BacktestRunner] {errors} orders skipped due to errors")

        if not rows:
            return ""

        df = pd.DataFrame(rows).sort_values(
            ["timestamp", "instrument_id", "order_id"],
        )
        df.to_csv(filepath, index=False)
        print(f"[BacktestRunner] Exported {len(df)} fills → {filepath}")
        return filepath

    def export_positions(self, filepath: str) -> str:
        """导出持仓快照为 CSV。返回文件路径，无持仓快照时返回空字符串。"""
        import pandas as pd
        from datetime import timezone, timedelta

        # 定义北京时间时区（UTC+8）
        CST = timezone(timedelta(hours=8))

        # 从 bridge 获取持仓历史
        bridge = getattr(self, '_bridge', None)
        if bridge is None:
            return ""

        position_history = getattr(bridge, '_position_history', [])
        if not position_history:
            return ""

        # 展开持仓快照为表格
        rows = []
        for snapshot in position_history:
            ts = snapshot['timestamp']
            positions = snapshot['positions']
            prices = snapshot.get('prices', {})  # 获取价格字典

            # 如果持仓为空（表示日终平仓后持仓为0），添加一个特殊标记
            if not positions:
                dt_utc = pd.Timestamp(int(ts), unit='ns', tz='UTC')
                dt_cst = dt_utc.tz_convert(CST)
                rows.append({
                    "timestamp": dt_cst.strftime('%Y-%m-%d %H:%M:%S'),
                    "instrument_id": "__DAY_END__",
                    "qty": 0.0,
                    "price": 0.0,
                })
            else:
                for code, qty in positions.items():
                    if abs(qty) > 1e-8:
                        dt_utc = pd.Timestamp(int(ts), unit='ns', tz='UTC')
                        dt_cst = dt_utc.tz_convert(CST)
                        price = prices.get(code, 0.0)  # 获取该合约的价格
                        rows.append({
                            "timestamp": dt_cst.strftime('%Y-%m-%d %H:%M:%S'),
                            "instrument_id": code,
                            "qty": qty,
                            "price": price,
                        })

        if not rows:
            return ""

        df = pd.DataFrame(rows).sort_values(
            ["timestamp", "instrument_id"],
        )
        df.to_csv(filepath, index=False)
        print(f"[BacktestRunner] Exported {len(df)} position snapshots → {filepath}")
        return filepath

    def _build_equity_from_fills(self, initial: float) -> pd.Series:
        """按时间顺序回放成交，构造逐交易日盯市权益。"""
        import pandas as pd
        import os
        from collections import deque, defaultdict
        from zoneinfo import ZoneInfo

        orders = getattr(self, '_cached_orders', [])
        if not orders:
            return pd.Series(dtype=float)

        sorted_orders = sorted(
            [o for o in orders if hasattr(o, 'filled_qty')],
            key=_order_fill_timestamp_ns,
        )

        def trading_day(order):
            ts = _order_fill_timestamp_ns(order)
            return pd.Timestamp(
                ts,
                unit='ns',
                tz='UTC',
            ).tz_convert(ZoneInfo("Asia/Shanghai")).strftime('%Y-%m-%d')

        # 先收集合约和成交日期，行情加载完成后再建立完整交易日序列。
        inst_set = set()
        fill_days = set()
        orders_by_day = defaultdict(list)
        for o in sorted_orders:
            try:
                inst = str(o.instrument_id).split('.')[0]
                day = trading_day(o)
                inst_set.add(inst)
                fill_days.add(day)
                orders_by_day[day].append(o)
            except Exception:
                continue
        if not fill_days:
            return pd.Series(dtype=float)
        min_day = min(fill_days)
        max_day = max(fill_days)
        if self._period_end:
            max_day = max(max_day, pd.Timestamp(self._period_end).strftime("%Y-%m-%d"))

        # 加载每个交易日收盘价
        # 优先从 _daily 文件加载（已按 trade_date 聚合，对齐交易所口径）
        # 回退到分钟线文件（取每日最后一根 bar）
        # 期权特殊处理: opt_daily 不存在时优先回退 opt_5min（数据更完整），再回退 opt_1min
        daily_close = defaultdict(dict)
        pq_instruments = defaultdict(set)
        for inst in inst_set:
            daily_pq = _parquet_for_inst(inst, "D1")
            if os.path.exists(os.path.join(self._data_dir, daily_pq)):
                pq_instruments[daily_pq].add(inst)
            elif is_option(inst):
                # 期权: 优先 5min（数据覆盖到最新），再回退 1min
                pq5 = _parquet_for_inst(inst, "M5")
                if os.path.exists(os.path.join(self._data_dir, pq5)):
                    pq_instruments[pq5].add(inst)
                else:
                    pq_instruments[_parquet_for_inst(inst, "M1")].add(inst)
            else:
                pq_instruments[_parquet_for_inst(inst, "M1")].add(inst)
        for pq_name, codes in pq_instruments.items():
            pq = os.path.join(self._data_dir, pq_name)
            if not os.path.exists(pq):
                continue
            try:
                cols = ['code', 'timestamp', 'close']
                # daily 文件可能有 trade_date 列
                try:
                    df = pd.read_parquet(pq, columns=cols + ['trade_date'],
                                        filters=[('code', 'in', list(codes))])
                    # 如果有 trade_date，用它作为日期键（对齐交易所口径）
                    if 'trade_date' in df.columns:
                        df['date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
                        daily = df.groupby(['date', 'code'])['close'].last()
                    else:
                        df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
                        daily = df.groupby(['date', 'code'])['close'].last()
                except Exception:
                    df = pd.read_parquet(pq, columns=cols,
                                        filters=[('code', 'in', list(codes))])
                    if df.empty:
                        continue
                    df['date'] = df['timestamp'].dt.strftime('%Y-%m-%d')
                    daily = df.groupby(['date', 'code'])['close'].last()
                if df.empty:
                    continue
                for (day, code), close in daily.items():
                    if min_day <= day <= max_day:
                        daily_close[day][code] = float(close)
            except Exception as exc:
                print(f"[BacktestRunner] 无法加载 {pq_name} 的日收盘价: {exc}")

        # 期权到期信息：{code: (exp_day, settlePrice)}
        option_expiry = {}
        metadata_dir = os.environ.get("BOMBER_METADATA_DIR") or os.path.join(self._data_dir, 'metadata')
        settlement_path = os.path.join(metadata_dir, 'opt_eod.parquet')
        try:
            eod_df = pd.read_parquet(
                settlement_path,
                columns=['tradeID','TradeDate','lastTradingDate','settlePrice'])
            eod_df = eod_df.dropna()
            eod_df['TradeDate_d'] = pd.to_datetime(eod_df['TradeDate']).dt.strftime('%Y-%m-%d')
            eod_df['lastTradingDate_d'] = pd.to_datetime(eod_df['lastTradingDate']).dt.strftime('%Y-%m-%d')
            expiry = eod_df[eod_df['TradeDate_d'] == eod_df['lastTradingDate_d']]
            for _, row in expiry.iterrows():
                code = str(row['tradeID']).strip()
                sp = float(row['settlePrice'])
                exp_day = row['lastTradingDate_d']
                if code in inst_set and sp >= 0:      # 允许 0：OTM 期权到期归零
                    daily_close[exp_day][code] = sp
                    option_expiry[code] = (exp_day, sp)
        except Exception as exc:
            if self._strict_data_validation and any(
                is_option(inst) for inst in inst_set
            ):
                raise RuntimeError(
                    f"Unable to load option settlement data: {settlement_path}"
                ) from exc
            print(f"[BacktestRunner] 无法加载期权结算数据: {exc}")

        expected_expiry_days = {
            day
            for inst, day in self._option_expiry_dates.items()
            if inst in inst_set and min_day <= day <= max_day
        }
        all_days = sorted(set(daily_close) | fill_days | expected_expiry_days)
        if not all_days:
            return pd.Series(dtype=float)

        # 构建到期日结算价映射 {inst: (exp_day, settle_price)}，供到期日填充使用
        expiry_settle_price = {}
        for inst, (exp_day, sp) in option_expiry.items():
            expiry_settle_price[(inst, exp_day)] = sp

        # 必须逐日应用成交，再用”当日持仓”盯市。不能先构造最终持仓。
        pos = {}
        eq_data = {}
        cum_realized = 0.0
        last_close = {}
        for day in all_days:
            for o in orders_by_day.get(day, []):
                try:
                    qv = o.filled_qty
                    fq = float(qv.as_double()) if hasattr(qv, 'as_double') else float(qv)
                    if fq <= 0:
                        continue
                    fp = float(o.avg_px.as_double()) if hasattr(o.avg_px, 'as_double') else float(o.avg_px)
                    inst = str(o.instrument_id).split('.')[0]
                    mul = self._instrument_multiplier(inst)
                    # 到期日期权：用交易所结算价计算平仓盈亏（替代市场成交价）
                    # 开仓仍以市场价入 FIFO 队列，由下方 option_expiry 代码以结算价收尾
                    fp_for_pnl = fp
                    if is_option(inst):
                        sp_key = (inst, day)
                        if sp_key in expiry_settle_price:
                            fp_for_pnl = expiry_settle_price[sp_key]
                    if inst not in pos:
                        pos[inst] = (deque(), deque())
                    long_queue, short_queue = pos[inst]
                    rem = fq
                    pnl = 0.0
                    if _order_is_buy(o):
                        while rem > 0 and short_queue:
                            open_qty, open_px = short_queue[0]
                            matched = min(rem, open_qty)
                            pnl += (open_px - fp_for_pnl) * matched * mul
                            rem -= matched
                            if open_qty <= matched:
                                short_queue.popleft()
                            else:
                                short_queue[0] = (open_qty - matched, open_px)
                        if rem > 0:
                            long_queue.append((rem, fp))
                    else:
                        while rem > 0 and long_queue:
                            open_qty, open_px = long_queue[0]
                            matched = min(rem, open_qty)
                            pnl += (fp_for_pnl - open_px) * matched * mul
                            rem -= matched
                            if open_qty <= matched:
                                long_queue.popleft()
                            else:
                                long_queue[0] = (open_qty - matched, open_px)
                        if rem > 0:
                            short_queue.append((rem, fp))
                    cum_realized += pnl
                    # 手续费：默认不计入权益（对齐第三方费前口径），仅统计输出
                    # 若 reporting.include_commission=true 才扣减
                    if self._include_commission:
                        if self._futures_bps > 0 or self._option_per_contract > 0:
                            cum_realized -= _compute_commission(
                                inst, fq, fp, mul,
                                self._futures_bps, self._option_per_contract,
                            )
                        else:
                            commissions = getattr(o, "commissions", None)
                            if callable(commissions):
                                for money in commissions():
                                    cum_realized -= (
                                        float(money.as_double())
                                        if hasattr(money, "as_double")
                                        else float(money)
                                    )
                except Exception as exc:
                    print(f"[BacktestRunner] 跳过无法回放的订单: {exc}")

            # 期权到期：按结算价强制平仓 FIFO 队列，计入已实现盈亏，队列清零
            if self._strict_data_validation:
                for inst, exp_day in self._option_expiry_dates.items():
                    if exp_day != day or inst not in pos or inst in option_expiry:
                        continue
                    long_queue, short_queue = pos[inst]
                    if long_queue or short_queue:
                        raise ValueError(
                            f"Missing expiry settlement for open option {inst} on {day}",
                        )
            for inst, (exp_day, sp) in option_expiry.items():
                if exp_day != day or inst not in pos:
                    continue
                long_queue, short_queue = pos[inst]
                mul = self._instrument_multiplier(inst)
                while long_queue:
                    qty, open_px = long_queue.popleft()
                    cum_realized += (sp - open_px) * qty * mul
                while short_queue:
                    qty, open_px = short_queue.popleft()
                    cum_realized += (open_px - sp) * qty * mul

            unrealized = 0.0
            closes = daily_close.get(day, {})
            last_close.update(closes)
            for inst, (lq, sq) in pos.items():
                c = last_close.get(inst)
                is_opt = is_option(inst)
                # 期货收盘价必须 > 0；期权允许 = 0（理论上到期日队列已清空，不会走到这）
                if c is None or (c <= 0 and not is_opt): continue
                mul = self._instrument_multiplier(inst)
                for q, op in lq: unrealized += (c - op) * q * mul
                for q, op in sq: unrealized += (op - c) * q * mul
            eq_data[day] = initial + cum_realized + unrealized

        eq = pd.Series(eq_data)
        eq.index = pd.to_datetime(eq.index)
        if isinstance(eq.index, pd.DatetimeIndex):
            eq = eq.sort_index().resample('D').last().ffill().dropna()
        return eq


def _parquet_bar_generator(
    data_dir: str,
    instruments: dict,
    bar_periods: dict,
    period_start: str = None,
    period_end: str = None,
    venue: str = "CFFEX",
    bar_timestamp: str = "end",
    batch_size: int = 5000,
    data_loader=None,  # 新增：允许传递已有的 data_loader 以使用缓存
):
    """按合约一次性加载全量 bars，内存排序后分批 yield（保序 + 高吞吐）"""
    # 优化：使用已有的 data_loader 实例以利用缓存，避免重复 IO
    if data_loader is None:
        from bomber_adapter.data_loader import FileDataLoader
        loader = FileDataLoader(
            data_dir,
            venue,
            bar_timestamp=bar_timestamp,
        )
    else:
        loader = data_loader

    inst_ids = list(instruments.keys())

    # 按 parquet 文件分组批量加载（每个文件只读一次），内存排序后分批 yield
    inst_periods = {inst_id: bar_periods.get(inst_id, "M1") for inst_id in inst_ids}
    bars_by_inst = loader.load_bars_batch(
        inst_periods, instruments,
        period_start=period_start, period_end=period_end)

    all_bars = []
    for inst_id, bars in bars_by_inst.items():
        if bars:
            all_bars.extend(bars)

    if not all_bars:
        return
    all_bars.sort(key=lambda b: b.ts_event)
    for i in range(0, len(all_bars), batch_size):
        batch = all_bars[i:i + batch_size]
        yield batch
    del all_bars


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


def _load_option_settlement(data_dir: str) -> dict:
    """从 data/metadata/opt_eod.parquet 加载期权到期日结算价，返回 {inst_id: price}"""
    import os
    import pandas as pd
    metadata_dir = os.environ.get("BOMBER_METADATA_DIR") or os.path.join(data_dir, "metadata")
    pq = os.path.join(metadata_dir, "opt_eod.parquet")
    if not os.path.exists(pq):
        return {}
    try:
        df = pd.read_parquet(pq, columns=["tradeID", "TradeDate", "lastTradingDate", "settlePrice"])
        df = df.dropna(subset=["tradeID", "TradeDate", "lastTradingDate", "settlePrice"])
        df["TradeDate"] = pd.to_datetime(df["TradeDate"])
        df["lastTradingDate"] = pd.to_datetime(df["lastTradingDate"])
        # 只在最后交易日取结算价
        df = df[df["TradeDate"] == df["lastTradingDate"]]
        result = {}
        for _, row in df.iterrows():
            tid = str(row["tradeID"]).strip()
            sp = float(row["settlePrice"])
            if not tid or sp <= 0:
                continue
            result[tid] = sp
        print(f"[BacktestRunner] 加载 {len(result)} 个期权结算价")
        return result
    except Exception as e:
        print(f"[BacktestRunner] 加载期权结算价失败: {e}")
        return {}


def _add_option_settlement(pos: dict, eq_data: dict, initial: float, cum_cash: float):
    """用 opt_eod 结算价关闭未平仓期权，补偿到期日权益"""
    import pandas as pd, os
    from collections import deque, defaultdict

    data_dir = os.environ.get("BOMBER_DATA_DIR") or os.environ.get("BT_DATA_DIR", "")
    if not data_dir:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    metadata_dir = os.environ.get("BOMBER_METADATA_DIR") or os.path.join(data_dir, "metadata")
    pq = os.path.join(metadata_dir, "opt_eod.parquet")
    if not os.path.exists(pq):
        print("[WARNING] metadata/opt_eod.parquet 不存在，期权到期未结算")
        return
    try:
        eod = pd.read_parquet(pq, columns=["tradeID", "lastTradingDate", "settlePrice"])
    except Exception as e:
        print(f"[WARNING] metadata/opt_eod.parquet 读取失败: {e}")
        return

    eod = eod.dropna(subset=["lastTradingDate", "settlePrice"])
    eod["lastTradingDate"] = pd.to_datetime(eod["lastTradingDate"])
    settle_map = {}
    for _, row in eod.iterrows():
        tid = str(row["tradeID"]).strip()
        if tid:
            settle_map[tid] = (float(row["settlePrice"]), 100.0,
                               row["lastTradingDate"].strftime("%Y-%m-%d"))

    settle_pnl = defaultdict(float)
    for inst, (long_q, short_q) in pos.items():
        if inst not in settle_map:
            continue
        sp, mul, exp_date = settle_map[inst]
        while long_q:
            qty, open_px = long_q.popleft()
            settle_pnl[exp_date] += (sp - open_px) * qty * mul
        while short_q:
            qty, open_px = short_q.popleft()
            settle_pnl[exp_date] += (open_px - sp) * qty * mul

    # 结算 PnL 追加到 eq_data
    last_equity = eq_data[max(eq_data.keys())] if eq_data else initial + cum_cash
    for exp_date, pnl in sorted(settle_pnl.items()):
        if exp_date not in eq_data:
            eq_data[exp_date] = last_equity
        eq_data[exp_date] += pnl
        last_equity = eq_data[exp_date]
