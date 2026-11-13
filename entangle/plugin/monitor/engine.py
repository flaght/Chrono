# -*- coding: utf-8 -*-
"""统一资金、持仓、委托、成交和持仓明细的监控插件。"""

import threading
import time
from collections import defaultdict

import pandas as pd
from vnpy.trader.event import (
    EVENT_ACCOUNT,
    EVENT_ORDER,
    EVENT_POSITION,
    EVENT_TRADE,
    EVENT_TIMER,
)

from .ctp_adapter import EVENT_CTP_QUERY_RESPONSE, install_query_callbacks
from .scheduler import QueryScheduler
from .schema import NORMALIZERS, TABLE_COLUMNS, normalize_fund, normalize_order
from .schema import normalize_position, normalize_trade, row_key
from .storage import Storage


QUERY_TYPES = (
    "funds", "positions", "orders", "trades", "position_details"
)


class Engine(object):
    """共享 Monitor 现有 CTP 网关的统一监控插件。"""

    def __init__(self, main_engine, event_engine, account_configs,
                 gateway_module, storage_adapter=None,
                 min_query_interval=1.1, query_intervals=None,
                 market_time_checker=None, trading_day_resolver=None):
        self.main_engine = main_engine
        self.event_engine = event_engine
        self.account_configs = account_configs
        self.gateway_module = gateway_module
        self.storage = Storage(storage_adapter)
        self.market_time_checker = market_time_checker
        self.trading_day_resolver = trading_day_resolver
        self.query_intervals = query_intervals or {
            "funds": 60.0,
            "positions": 60.0,
            "orders": 300.0,
            "trades": 300.0,
            "position_details": 300.0,
        }

        install_query_callbacks(gateway_module)

        # table_name -> gateway_name -> unique_key -> normalized row
        self._tables = {
            name: defaultdict(dict) for name in QUERY_TYPES
        }
        self.funds = self._tables["funds"]
        self.positions = self._tables["positions"]
        self.orders = self._tables["orders"]
        self.trades = self._tables["trades"]
        self.position_details = self._tables["position_details"]
        self._lock = threading.RLock()
        self._started = False
        self._stop_event = threading.Event()
        self._periodic_thread = None
        self._next_due = {}
        self._last_trade_refresh = defaultdict(float)
        self._last_fund_snapshot_time = None
        self._last_order_snapshot_state = None
        self._last_trade_snapshot_state = None

        self.scheduler = QueryScheduler(
            main_engine=main_engine,
            account_configs=account_configs,
            result_handler=self._apply_query_result,
            min_interval=min_query_interval,
        )

        self.event_engine.register(
            EVENT_CTP_QUERY_RESPONSE, self.process_query_response
        )
        self.event_engine.register(EVENT_ACCOUNT, self.process_account_event)
        self.event_engine.register(EVENT_POSITION, self.process_position_event)
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)

    def start(self):
        """接管周期查询，并在现有连接上建立五张表的初始快照。"""
        if self._started:
            return
        self._started = True
        self._stop_event.clear()

        # vnpy_ctp 默认定时器会独立查询资金/持仓。停用它，避免和统一队列抢流控。
        for gateway_name in list(self.account_configs.keys()):
            self.attach_gateway(gateway_name)

        self.scheduler.start()
        now = time.time()
        for query_type, interval in self.query_intervals.items():
            self._next_due[query_type] = now + float(interval)

        self.request_full_snapshot(wait=False)
        self._periodic_thread = threading.Thread(
            target=self._periodic_loop,
            name="ctp-monitor-periodic",
            daemon=True,
        )
        self._periodic_thread.start()

    def attach_gateway(self, gateway_name):
        """让指定网关停用 vnpy_ctp 自带轮询，改由统一队列调度。"""
        gateway = self.main_engine.get_gateway(gateway_name)
        if gateway and hasattr(gateway, "process_timer_event"):
            try:
                self.event_engine.unregister(
                    EVENT_TIMER, gateway.process_timer_event
                )
            except Exception:
                pass

    def stop(self):
        if not self._started:
            return
        self._started = False
        self._stop_event.set()
        if self._periodic_thread:
            self._periodic_thread.join(timeout=3.0)
        self.scheduler.stop()

    def request_full_snapshot(self, gateway_name=None, wait=False,
                              timeout=30.0, ready_timeout=120.0):
        """请求一个或全部账户的五类柜台快照。"""
        gateway_names = (
            [gateway_name] if gateway_name
            else list(self.account_configs.keys())
        )
        tasks = []
        for name in gateway_names:
            for query_type in QUERY_TYPES:
                tasks.append(self.scheduler.request(
                    name,
                    query_type,
                    timeout=timeout,
                    ready_timeout=ready_timeout,
                ))
        if wait:
            for task in tasks:
                if not task.completed.wait(ready_timeout + timeout + 5.0):
                    raise TimeoutError(
                        "等待 {0} {1} 查询任务超时".format(
                            task.gateway_name, task.query_type
                        )
                    )
                if task.error:
                    raise RuntimeError(
                        "{0} {1}: {2}".format(
                            task.gateway_name, task.query_type, task.error
                        )
                    )
        return tasks

    def request_tables(self, table_names, gateway_name=None, wait=False,
                       timeout=30.0, ready_timeout=120.0):
        gateway_names = (
            [gateway_name] if gateway_name
            else list(self.account_configs.keys())
        )
        tasks = []
        for name in gateway_names:
            for table_name in table_names:
                tasks.append(self.scheduler.request(
                    name, table_name, timeout, ready_timeout
                ))
        if wait:
            for task in tasks:
                if not task.completed.wait(ready_timeout + timeout + 5.0):
                    raise TimeoutError(
                        "等待 {0} {1} 查询任务超时".format(
                            task.gateway_name, task.query_type
                        )
                    )
                if task.error:
                    raise RuntimeError(task.error)
        return tasks

    def process_query_response(self, event):
        self.scheduler.handle_response(event.data)

    def _apply_query_result(self, task):
        config = self.account_configs[task.gateway_name]
        normalizer = NORMALIZERS[task.query_type]
        rows = [
            normalizer(raw, task.gateway_name, config)
            for raw in task.rows
        ]

        # CTP 持仓查询可能按今昨仓返回多行，合并为合约+方向级别。
        if task.query_type == "positions":
            rows = self._aggregate_positions(rows)

        snapshot = {}
        for row in rows:
            snapshot[row_key(task.query_type, row)] = row

        with self._lock:
            self._tables[task.query_type][task.gateway_name] = snapshot
        print("[{0}] {1} 查询完成，共 {2} 条".format(
            task.gateway_name, task.query_type, len(snapshot)
        ))

    def _aggregate_positions(self, rows):
        result = {}
        sum_fields = (
            "Position", "YdPosition", "TodayPosition", "LongFrozen",
            "ShortFrozen", "OpenVolume", "CloseVolume", "OpenAmount",
            "CloseAmount", "PositionCost", "UseMargin", "Commission",
            "CloseProfit", "PositionProfit",
        )
        for row in rows:
            key = "|".join([
                str(row.get("GatewayName", "")),
                str(row.get("ExchangeID", "")),
                str(row.get("InstrumentID", "")),
                str(row.get("PosiDirection", "")),
            ])
            if key not in result:
                result[key] = dict(row)
                continue
            target = result[key]
            for field in sum_fields:
                target[field] = (
                    float(target.get(field, 0) or 0)
                    + float(row.get(field, 0) or 0)
                )
            target["PositionDate"] = ""
        return list(result.values())

    def _config_for(self, gateway_name):
        return self.account_configs.get(gateway_name, {})

    def _upsert(self, table_name, gateway_name, row):
        with self._lock:
            self._tables[table_name][gateway_name][
                row_key(table_name, row)
            ] = row

    def process_account_event(self, event):
        account = event.data
        gateway_name = account.gateway_name
        raw = dict(getattr(account, "extra", {}) or {})
        raw.update({
            "AccountID": account.accountid,
            "Balance": account.balance,
            "Available": account.available,
            "FrozenMargin": getattr(account, "frozen_margin", account.frozen),
            "TradingDay": getattr(account, "trading_day", ""),
        })
        row = normalize_fund(raw, gateway_name, self._config_for(gateway_name))
        self._upsert("funds", gateway_name, row)

    def process_position_event(self, event):
        position = event.data
        gateway_name = position.gateway_name
        exchange = getattr(position.exchange, "value", position.exchange)
        direction = getattr(position.direction, "value", position.direction)
        raw = {
            "ExchangeID": exchange,
            "InstrumentID": position.symbol,
            "PosiDirection": direction,
            "Position": position.volume,
            "YdPosition": position.yd_volume,
            "TodayPosition": position.volume - position.yd_volume,
            "PositionProfit": getattr(position, "pnl", 0.0),
        }
        row = normalize_position(
            raw, gateway_name, self._config_for(gateway_name)
        )
        row["DirectionName"] = str(direction)
        key = row_key("positions", row)
        with self._lock:
            if position.volume == 0:
                self.positions[gateway_name].pop(key, None)
            else:
                self.positions[gateway_name][key] = row

    def process_order_event(self, event):
        order = event.data
        gateway_name = order.gateway_name
        exchange = getattr(order.exchange, "value", order.exchange)
        direction = getattr(order.direction, "value", order.direction)
        offset = getattr(order.offset, "value", order.offset)
        status = getattr(order.status, "value", order.status)
        front_id, session_id, order_ref = "", "", order.orderid
        parts = str(order.orderid).split("_")
        if len(parts) == 3:
            front_id, session_id, order_ref = parts
        raw = {
            "ExchangeID": exchange,
            "InstrumentID": order.symbol,
            "FrontID": front_id,
            "SessionID": session_id,
            "OrderRef": order_ref,
            "Direction": direction,
            "CombOffsetFlag": offset,
            "LimitPrice": order.price,
            "VolumeTotalOriginal": order.volume,
            "VolumeTraded": order.traded,
            "VolumeTotal": order.volume - order.traded,
            "OrderStatus": status,
        }
        row = normalize_order(raw, gateway_name, self._config_for(gateway_name))
        row["DirectionName"] = str(direction)
        row["OffsetName"] = str(offset)
        row["OrderStatusName"] = str(status)
        self._upsert("orders", gateway_name, row)

    def process_trade_event(self, event):
        trade = event.data
        gateway_name = trade.gateway_name
        exchange = getattr(trade.exchange, "value", trade.exchange)
        direction = getattr(trade.direction, "value", trade.direction)
        offset = getattr(trade.offset, "value", trade.offset)
        trade_dt = getattr(trade, "datetime", None)
        raw = dict(getattr(trade, "extra", {}) or {})
        raw.update({
            "ExchangeID": exchange,
            "InstrumentID": trade.symbol,
            "TradeID": trade.tradeid,
            "OrderSysID": (
                raw.get("OrderSysID")
                or getattr(trade, "sys_orderid", "")
            ),
            "OrderLocalID": (
                raw.get("OrderLocalID")
                or getattr(trade, "orderid", "")
            ),
            "Direction": direction,
            "OffsetFlag": offset,
            "Price": trade.price,
            "Volume": trade.volume,
            "TradeDate": trade_dt.strftime("%Y%m%d") if trade_dt else "",
            "TradeTime": trade_dt.strftime("%H:%M:%S") if trade_dt else "",
        })
        row = normalize_trade(raw, gateway_name, self._config_for(gateway_name))
        row["DirectionName"] = str(direction)
        row["OffsetName"] = str(offset)
        self._upsert("trades", gateway_name, row)

        # 成交后刷新持仓和持仓明细，5秒内多笔成交只安排一次。
        now = time.time()
        if now - self._last_trade_refresh[gateway_name] >= 5.0:
            self._last_trade_refresh[gateway_name] = now
            self.request_tables(
                ("positions", "position_details"), gateway_name=gateway_name
            )

    def get_dataframe(self, table_name):
        if table_name not in TABLE_COLUMNS:
            raise ValueError("未知监控表: {0}".format(table_name))
        with self._lock:
            rows = []
            for snapshot in self._tables[table_name].values():
                rows.extend(dict(row) for row in snapshot.values())
        return pd.DataFrame(rows, columns=TABLE_COLUMNS[table_name])

    def get_all_dataframes(self):
        return {
            name: self.get_dataframe(name) for name in QUERY_TYPES
        }

    def get_funds_dataframe(self):
        return self.get_dataframe("funds")

    def get_positions_dataframe(self):
        return self.get_dataframe("positions")

    def get_orders_dataframe(self):
        return self.get_dataframe("orders")

    def get_trades_dataframe(self):
        return self.get_dataframe("trades")

    def get_position_details_dataframe(self):
        return self.get_dataframe("position_details")

    def save_table(self, table_name, db_name, target_table=None,
                   method="increment"):
        dataframe = self.get_dataframe(table_name)
        return self.storage.save(
            dataframe,
            db_name,
            target_table or table_name,
            method=method,
        )

    def to_account_fund_snapshots(self, interval_minutes=5):
        """生成与 DolphinDB AccountFundSnapshots 表兼容的资金快照。"""
        funds = self.get_funds_dataframe()
        columns = [
            "UpdateTime", "TradingDay", "ProductID", "AccountID", "BrokerID",
            "Balance", "DynamicEquity", "Margin", "Available", "CloseProfit",
            "PositionProfit", "Commission", "FrozenMargin", "Deposit", "Withdraw",
        ]
        if funds.empty:
            return pd.DataFrame(columns=columns)

        now = self._china_now()
        if interval_minutes > 1:
            minute = (now.minute // interval_minutes) * interval_minutes
            update_time = now.replace(
                minute=minute, second=0, microsecond=0
            )
        else:
            update_time = now.replace(microsecond=0)

        fallback_day = (
            self.trading_day_resolver(now)
            if self.trading_day_resolver else now.strftime("%Y-%m-%d")
        )
        result = funds.copy()
        result["UpdateTime"] = update_time
        result["TradingDay"] = result["TradingDay"].apply(
            lambda value: self._format_trading_day(value, fallback_day)
        )
        return result[columns]

    def save_account_fund_snapshots(
            self, db_name="nurex_daily", table_name="AccountFundSnapshots",
            force=False, interval_minutes=5, stale_timeout=180.0):
        """由 monitor 插件写入 AccountFundSnapshots，包含时段、存活和防重检查。"""
        now = self._china_now()
        if (
            not force
            and self.market_time_checker
            and not self.market_time_checker(now)
        ):
            return False

        source = self.get_funds_dataframe()
        if source.empty:
            return False

        if not force:
            update_times = pd.to_datetime(
                source["UpdateTime"], errors="coerce"
            ).dropna()
            if update_times.empty:
                return False
            # 所有待写入账户都必须保持新鲜，避免混入某个已断开的账户快照。
            age = (now - update_times.min().to_pydatetime()).total_seconds()
            if age > stale_timeout:
                return False

        dataframe = self.to_account_fund_snapshots(interval_minutes)
        if dataframe.empty:
            return False
        update_time = dataframe["UpdateTime"].iloc[0]
        if not force and self._last_fund_snapshot_time == update_time:
            return False

        try:
            saved = self.storage.save(
                dataframe, db_name, table_name, method="increment"
            )
            if saved:
                self._last_fund_snapshot_time = update_time
                print(
                    "[{0}] monitor 写入 {1}.{2} 共 {3} 条 [Bar时间: {4}]".format(
                        now.strftime("%H:%M:%S"), db_name, table_name,
                        len(dataframe), update_time,
                    )
                )
            return saved
        except Exception as exc:
            print("[monitor DolphinDB入库异常] {0}".format(exc))
            return False

    def save_order_snapshots(
            self, db_name="nurex_daily", table_name="DerivativeOrder",
            force=False):
        """将当日委托快照写入 DolphinDB。

        委托数据由查询回报和 EVENT_ORDER 共同维护；这里只负责把
        当前统一快照落库，不会建立新的 CTP 连接。
        """
        now = self._china_now()
        if (
            not force
            and self.market_time_checker
            and not self.market_time_checker(now)
        ):
            return False

        dataframe = self.get_orders_dataframe()
        if dataframe.empty:
            return False

        # UpdateTime 是每条委托最后被监控插件更新的时间。行数与
        # 最新更新时间都未变时，避免主循环重复追加。
        update_times = pd.to_datetime(
            dataframe["UpdateTime"], errors="coerce"
        ).dropna()
        latest_update = None if update_times.empty else update_times.max()
        snapshot_state = (len(dataframe), latest_update)
        if not force and self._last_order_snapshot_state == snapshot_state:
            return False

        try:
            saved = self.storage.save(
                dataframe, db_name, table_name, method="increment"
            )
            if saved:
                self._last_order_snapshot_state = snapshot_state
                print(
                    "[{0}] monitor 写入 {1}.{2} 共 {3} 条".format(
                        now.strftime("%H:%M:%S"), db_name, table_name,
                        len(dataframe),
                    )
                )
            return saved
        except Exception as exc:
            print("[monitor 委托表入库异常] {0}".format(exc))
            return False

    def save_trade_snapshots(
            self, db_name="nurex_daily", table_name="DerivativeTrade",
            force=False):
        """将当日成交快照写入 DolphinDB。"""
        now = self._china_now()
        if (
            not force
            and self.market_time_checker
            and not self.market_time_checker(now)
        ):
            return False

        dataframe = self.get_trades_dataframe()
        if dataframe.empty:
            return False

        update_times = pd.to_datetime(
            dataframe["UpdateTime"], errors="coerce"
        ).dropna()
        latest_update = None if update_times.empty else update_times.max()
        snapshot_state = (len(dataframe), latest_update)
        if not force and self._last_trade_snapshot_state == snapshot_state:
            return False

        try:
            saved = self.storage.save(
                dataframe, db_name, table_name, method="increment"
            )
            if saved:
                self._last_trade_snapshot_state = snapshot_state
                print(
                    "[{0}] monitor 写入 {1}.{2} 共 {3} 条".format(
                        now.strftime("%H:%M:%S"), db_name, table_name,
                        len(dataframe),
                    )
                )
            return saved
        except Exception as exc:
            print("[monitor 成交表入库异常] {0}".format(exc))
            return False

    def save_monitor_snapshots(
            self, force=False, interval_minutes=5, stale_timeout=180.0):
        """统一写入 monitor 管理的监控表。

        各张表独立校验、独立防重和独立返回结果；单表无数据或写入
        失败不会阻断其他表。后续增加持仓明细时也统一从此入口扩展。
        """
        db_name = "nurex_daily"
        table_names = {
            "funds": "AccountFundSnapshots",
            "orders": "DerivativeOrder",
            "trades": "DerivativeTrade",
        }
        return {
            table_names["funds"]: self.save_account_fund_snapshots(
                db_name=db_name,
                table_name=table_names["funds"],
                force=force,
                interval_minutes=interval_minutes,
                stale_timeout=stale_timeout,
            ),
            table_names["orders"]: self.save_order_snapshots(
                db_name=db_name,
                table_name=table_names["orders"],
                force=force,
            ),
            table_names["trades"]: self.save_trade_snapshots(
                db_name=db_name,
                table_name=table_names["trades"],
                force=force,
            ),
        }

    @staticmethod
    def _format_trading_day(value, fallback):
        text = "" if value is None else str(value).strip()
        if len(text) == 8 and text.isdigit():
            text = "{0}-{1}-{2}".format(text[:4], text[4:6], text[6:])
        if len(text) != 10:
            return fallback
        return max(text, fallback)

    @staticmethod
    def _china_now():
        import datetime
        timezone = datetime.timezone(datetime.timedelta(hours=8))
        return datetime.datetime.now(timezone).replace(tzinfo=None)

    def print_dashboard(self):
        counts = {
            name: len(self.get_dataframe(name)) for name in QUERY_TYPES
        }
        print(
            "监控插件 | 资金:{funds} 持仓:{positions} 委托:{orders} "
            "成交:{trades} 持仓明细:{position_details}".format(**counts)
        )

    def _periodic_loop(self):
        while not self._stop_event.wait(1.0):
            now = time.time()
            for query_type, interval in self.query_intervals.items():
                if now < self._next_due.get(query_type, now):
                    continue
                for gateway_name in list(self.account_configs.keys()):
                    try:
                        self.scheduler.request(gateway_name, query_type)
                    except Exception as exc:
                        print("[{0}] 安排 {1} 查询失败: {2}".format(
                            gateway_name, query_type, exc
                        ))
                self._next_due[query_type] = now + float(interval)

