"""CTP 行情链路的分层验证脚本。

本文件从最底层向上验证 CTP 行情能力，每个测试只增加一层复杂度：

1. ``test1``：不连接网络，用 FakeRawApi 验证 Python Driver 的调用顺序；
2. ``test2``：加载真实 CTP 动态库，验证 Native API 可以启动和安全释放；
3. ``test3``：连接真实行情前置并登录，但不订阅行情；
4. ``test4``：登录、订阅并接收一份 CTP 原始字典行情；
5. ``test5``：使用正式 CtpLiveDataFeed，验证原始行情可转换成标准
   QuoteTick/TradeTick。

这组测试只覆盖 CTP MdApi 行情链路，不涉及 TraderApi、真实下单、仓位管理、
组合管理或策略信号。
"""
from __future__ import annotations

import argparse
import logging
import tempfile
import time
import os
import threading
from pathlib import Path
from typing import Any, Mapping
from dotenv import load_dotenv
from market.native.ctp.driver import CtpMdDriver, NativeCtpMdDriver, create_native_driver

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("CtpNativeLifecycleTest")



def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"缺少环境变量: {name}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _error_id(error: Mapping[str, Any] | None) -> int:
    return int((error or {}).get("ErrorID", 0) or 0)


def _error_text(error: Mapping[str, Any] | None) -> str:
    error = error or {}
    return f"{_error_id(error)} {error.get('ErrorMsg', '')}".strip()


class FakeRawApi:
    """NativeCtpMdDriver 的无网络替身。

    每个方法只记录调用，不加载 CTP 动态库。它让 test1 能准确判断 Driver 是否
    以正确参数和顺序调用底层 API，而不会把网络、账号和交易时段问题混进来。
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def createFtdcMdApi(self, flow_path: str, production_mode: bool) -> None:
        self.calls.append(("create", flow_path, production_mode))

    def registerFront(self, front: str) -> None:
        self.calls.append(("front", front))

    def init(self) -> None:
        self.calls.append(("init",))

    def exit(self) -> int:
        self.calls.append(("exit",))
        return 1

    def release(self) -> None:
        self.calls.append(("release",))

    def reqUserLogin(self, request: dict, request_id: int) -> int:
        self.calls.append(("login", request, request_id))
        return 0

    def subscribeMarketData(self, symbol: str) -> int:
        self.calls.append(("subscribe", symbol))
        return 0

    def unSubscribeMarketData(self, symbol: str) -> int:
        self.calls.append(("unsubscribe", symbol))
        return 0


class ProbeCallbacks:
    """test2 使用的最小回调接收器。

    test2 的目标只是生命周期；测试地址故意不可达，因此任何回调都只写日志，
    不把“没有连接成功”视为失败。
    """

    def on_front_connected(self) -> None:
        logger.info("unexpected connection to unreachable test front")

    def on_front_disconnected(self, reason: int) -> None:
        logger.info("front disconnected during probe: reason=%s", reason)

    def on_login_response(self, error: Mapping[str, Any] | None) -> None:
        logger.info("unexpected login response: %s", error)

    def on_subscribe_response(
        self,
        data: Mapping[str, Any] | None,
        error: Mapping[str, Any] | None,
    ) -> None:
        logger.info("unexpected subscribe response: data=%s error=%s", data, error)

    def on_unsubscribe_response(
        self,
        data: Mapping[str, Any] | None,
        error: Mapping[str, Any] | None,
    ) -> None:
        logger.info("unexpected unsubscribe response: data=%s error=%s", data, error)

    def on_depth_market_data(self, data: Mapping[str, Any]) -> None:
        logger.info("unexpected market data: %s", data)

    def on_api_error(self, error: Mapping[str, Any] | None) -> None:
        logger.info("API error during probe: %s", error)


class LoginProbe:
    """把 CTP 异步连接和登录回调转换成可等待的测试状态。

    ``connected`` 表示收到 OnFrontConnected；``login_done`` 表示收到登录结果。
    两者分开可以区分“网络没有连上”和“连上但登录没有返回”。
    """

    def __init__(self, broker_id: str, user_id: str, password: str) -> None:
        self.broker_id = broker_id
        self.user_id = user_id
        self.password = password
        self.driver: CtpMdDriver | None = None
        self.connected = threading.Event()
        self.login_done = threading.Event()
        self.login_error: Mapping[str, Any] | None = None

    def on_front_connected(self) -> None:
        logger.info("CTP 行情前置连接成功，提交登录")
        self.connected.set()
        if self.driver is None:
            self.login_error = {"ErrorID": -1, "ErrorMsg": "driver is not assigned"}
            self.login_done.set()
            return
        try:
            self.driver.login(self.broker_id, self.user_id, self.password)
        except Exception as exc:
            self.login_error = {"ErrorID": -1, "ErrorMsg": str(exc)}
            self.login_done.set()

    def on_front_disconnected(self, reason: int) -> None:
        logger.warning("CTP 行情前置断开: reason=%s (0x%x)", reason, reason)

    def on_login_response(self, error: Mapping[str, Any] | None) -> None:
        self.login_error = error
        if _error_id(error):
            logger.error("CTP 行情登录失败: %s", _error_text(error))
        else:
            logger.info("CTP 行情登录成功")
        self.login_done.set()

    def on_subscribe_response(self, data, error) -> None:
        logger.warning("登录探针收到意外订阅响应: data=%s error=%s", data, error)

    def on_unsubscribe_response(self, data, error) -> None:
        logger.warning("登录探针收到意外退订响应: data=%s error=%s", data, error)

    def on_depth_market_data(self, data) -> None:
        logger.warning("登录探针收到意外行情: %s", data.get("InstrumentID", ""))

    def on_api_error(self, error: Mapping[str, Any] | None) -> None:
        logger.error("CTP API 错误: %s", _error_text(error))


class MarketDataProbe(LoginProbe):
    """在 LoginProbe 基础上增加订阅完成和第一份原始行情状态。"""

    def __init__(self, broker_id: str, user_id: str, password: str, symbol: str) -> None:
        super().__init__(broker_id, user_id, password)
        self.symbol = symbol
        self.subscribe_done = threading.Event()
        self.subscribe_error: Mapping[str, Any] | None = None
        self.tick_received = threading.Event()
        self.first_tick: dict[str, Any] | None = None
        self.latest_tick: dict[str, Any] | None = None
        self.tick_count = 0

    def on_login_response(self, error: Mapping[str, Any] | None) -> None:
        super().on_login_response(error)
        if _error_id(error) or self.driver is None:
            return
        try:
            logger.info("提交 CTP 行情订阅: %s", self.symbol)
            self.driver.subscribe(self.symbol)
        except Exception as exc:
            self.subscribe_error = {"ErrorID": -1, "ErrorMsg": str(exc)}
            self.subscribe_done.set()

    def on_subscribe_response(
        self,
        data: Mapping[str, Any] | None,
        error: Mapping[str, Any] | None,
    ) -> None:
        self.subscribe_error = error
        if _error_id(error):
            logger.error("CTP 行情订阅失败: %s", _error_text(error))
        else:
            logger.info("CTP 行情订阅成功: %s", (data or {}).get("InstrumentID", ""))
        self.subscribe_done.set()

    def on_depth_market_data(self, data: Mapping[str, Any]) -> None:
        self.latest_tick = dict(data)
        self.tick_count += 1
        if self.first_tick is not None:
            return
        self.first_tick = dict(data)
        logger.info(
            "收到第一份原始行情: instrument=%s trading_day=%s action_day=%s "
            "time=%s.%03d last=%s "
            "bid=%s@%s ask=%s@%s limit=[%s,%s] volume=%s open_interest=%s",
            data.get("InstrumentID", ""),
            data.get("TradingDay", ""),
            data.get("ActionDay", ""),
            data.get("UpdateTime", ""),
            int(data.get("UpdateMillisec", 0) or 0),
            data.get("LastPrice"),
            data.get("BidPrice1"),
            data.get("BidVolume1"),
            data.get("AskPrice1"),
            data.get("AskVolume1"),
            data.get("LowerLimitPrice"),
            data.get("UpperLimitPrice"),
            data.get("Volume"),
            data.get("OpenInterest"),
        )
        self.tick_received.set()



def test1() -> None:
    """阶段1：离线验证 Driver 到原始 API 的调用契约。

    验证内容：创建API、注册前置、初始化、登录请求号、订阅、退订和退出顺序。
    通过只说明 Python Driver 封装正确，不说明动态库可加载或网络可连接。
    """

    # FakeRawApi 不执行任何原生代码，只保留完整调用记录。
    raw = FakeRawApi()
    driver = NativeCtpMdDriver(raw)
    driver.start("tcp://test:1234", Path("/tmp/bomber-ctp-driver-test"), False)
    driver.login("9999", "demo", "secret")
    driver.subscribe("rb2610")
    driver.unsubscribe("rb2610")
    driver.stop()

    # start() 应先创建 API，再注册前置，最后 init() 启动 CTP 工作线程。
    assert raw.calls[0] == ("create", "/tmp/bomber-ctp-driver-test/", False)
    assert raw.calls[1] == ("front", "tcp://test:1234")
    assert raw.calls[2] == ("init",)
    # 第一次登录请求的 request_id 应从1开始。
    assert raw.calls[3][0] == "login"
    assert raw.calls[3][2] == 1
    assert raw.calls[-3:] == [
        ("subscribe", "rb2610"),
        ("unsubscribe", "rb2610"),
        ("exit",),
    ]
    print("CTP driver OK")



def test2():
    """阶段2：验证真实原生扩展和 CTP API 的启动/停止生命周期。

    使用不可达的本机端口，目的不是连接成功，而是确认动态库可加载、API可创建、
    后台线程可以启动，并且 stop() 不阻塞且能安全释放资源。
    """

    driver: CtpMdDriver = create_native_driver(ProbeCallbacks())
    with tempfile.TemporaryDirectory(prefix="bomber-ctp-lifecycle-") as directory:
        logger.info("starting CTP native API with unreachable local test front")
        started_at = time.monotonic()
        try:
            driver.start(
                front="tcp://127.0.0.1:1",
                flow_path=Path(directory),
                production_mode=False,
            )
            logger.info("CTP native API started")
            time.sleep(1.0)
        finally:
            logger.info("stopping CTP native API")
            driver.stop()
        elapsed = time.monotonic() - started_at

    logger.info("CTP native API stopped in %.3fs", elapsed)
    print("CTP native lifecycle OK")



def test3() -> None:
    """阶段3：验证真实前置连接和账号登录。

    验证 ``CTP_MD_ADDRESS`` 可达、production_mode 与服务器匹配、账号信息有效，
    并区分连接超时与登录错误。这里尚未订阅合约，也没有验证行情推送。
    """

    front = _required_env("CTP_MD_ADDRESS")
    broker_id = _required_env("CTP_BROKER_ID")
    user_id = _required_env("CTP_ACCOUNT_ID")
    password = _required_env("CTP_PASSWORD")
    timeout = float(os.getenv("CTP_CONNECT_TIMEOUT", "15"))
    production_mode = _env_bool("CTP_PRODUCTION_MODE", False)
    flow_path = Path(os.getenv("CTP_MD_FLOW_PATH", "/tmp/bomber-ctp-login-probe"))

    logger.info(
        "启动 CTP 登录探针: front=%s broker=%s user=%s production_mode=%s",
        front,
        broker_id,
        user_id,
        production_mode,
    )
    probe = LoginProbe(broker_id, user_id, password)
    driver = create_native_driver(probe)
    probe.driver = driver
    try:
        driver.start(front, flow_path, production_mode)
        # 先等待网络连接事件，再等待登录响应，便于准确定位失败阶段。
        if not probe.connected.wait(timeout):
            raise TimeoutError(f"{timeout:g} 秒内未连接到 CTP 行情前置")
        if not probe.login_done.wait(timeout):
            raise TimeoutError(f"连接成功，但 {timeout:g} 秒内未收到登录响应")
        if _error_id(probe.login_error):
            raise RuntimeError(f"CTP 行情登录失败: {_error_text(probe.login_error)}")
        print("CTP login OK")
    finally:
        logger.info("停止 CTP 登录探针")
        driver.stop()


def test4() -> None:
    """阶段4：验证原始 CTP DepthMarketData 订阅链路。

    登录成功后订阅 ``CTP_SYMBOL``，等待订阅响应和第一份原始行情字典。通过说明
    CTP服务端确实在推送该合约；仍未验证字段转换、精度或标准事件模型。
    """

    front = _required_env("CTP_MD_ADDRESS")
    broker_id = _required_env("CTP_BROKER_ID")
    user_id = _required_env("CTP_ACCOUNT_ID")
    password = _required_env("CTP_PASSWORD")
    symbol = _required_env("CTP_SYMBOL")
    connect_timeout = float(os.getenv("CTP_CONNECT_TIMEOUT", "15"))
    market_timeout = float(os.getenv("CTP_MARKET_TIMEOUT", "30"))
    sample_seconds = float(os.getenv("CTP_MARKET_SAMPLE_SECONDS", "0"))
    if sample_seconds < 0:
        raise ValueError("CTP_MARKET_SAMPLE_SECONDS不能为负数")
    production_mode = _env_bool("CTP_PRODUCTION_MODE", False)
    flow_path = Path(os.getenv("CTP_MD_FLOW_PATH", "/tmp/bomber-ctp-market-probe"))

    logger.info(
        "启动 CTP 原始行情探针: front=%s broker=%s user=%s symbol=%s production_mode=%s",
        front,
        broker_id,
        user_id,
        symbol,
        production_mode,
    )
    probe = MarketDataProbe(broker_id, user_id, password, symbol)
    driver = create_native_driver(probe)
    probe.driver = driver
    subscribed = False
    try:
        driver.start(front, flow_path, production_mode)
        if not probe.connected.wait(connect_timeout):
            raise TimeoutError(f"{connect_timeout:g} 秒内未连接到 CTP 行情前置")
        if not probe.login_done.wait(connect_timeout):
            raise TimeoutError(f"连接成功，但 {connect_timeout:g} 秒内未收到登录响应")
        if _error_id(probe.login_error):
            raise RuntimeError(f"CTP 行情登录失败: {_error_text(probe.login_error)}")
        # 订阅请求被调用不代表服务器接受，必须等待订阅响应并检查 ErrorID。
        if not probe.subscribe_done.wait(connect_timeout):
            raise TimeoutError(f"登录成功，但 {connect_timeout:g} 秒内未收到订阅响应")
        if _error_id(probe.subscribe_error):
            raise RuntimeError(f"CTP 行情订阅失败: {_error_text(probe.subscribe_error)}")
        subscribed = True
        # 非交易时段可能订阅成功却没有行情，因此使用独立 market_timeout。
        if not probe.tick_received.wait(market_timeout):
            raise TimeoutError(
                f"订阅成功，但 {market_timeout:g} 秒内未收到 {symbol} 行情；"
                "请确认交易时段和合约代码",
            )
        if sample_seconds:
            time.sleep(sample_seconds)
            latest = probe.latest_tick or {}
            logger.info(
                "采样末原始行情: ticks=%s trading_day=%s action_day=%s "
                "time=%s.%03d bid=%s@%s ask=%s@%s limit=[%s,%s]",
                probe.tick_count,
                latest.get("TradingDay", ""),
                latest.get("ActionDay", ""),
                latest.get("UpdateTime", ""),
                int(latest.get("UpdateMillisec", 0) or 0),
                latest.get("BidPrice1"), latest.get("BidVolume1"),
                latest.get("AskPrice1"), latest.get("AskVolume1"),
                latest.get("LowerLimitPrice"), latest.get("UpperLimitPrice"),
            )
        print("CTP raw market data OK")
    finally:
        if subscribed:
            try:
                driver.unsubscribe(symbol)
            except Exception:
                logger.exception("CTP 探针退订失败")
        logger.info("停止 CTP 原始行情探针")
        driver.stop()


def test5() -> None:
    """阶段5：验证正式 CTP Feed 输出统一标准行情。

    完整路径为：MdApi原始结构 → Native Driver → CTP Converter →
    CtpLiveDataFeed → QuoteTick/TradeTick 回调。

    QuoteTick 检查买卖数量精度一致，防止底层 Rust Quote 构造发生 panic；TradeTick
    检查价格和本次成交增量均为正。测试只接收行情，不会真实下单。
    """

    from decimal import Decimal

    from market.basic.base import DataType, InstrumentId, InstrumentMeta, QuoteTick, TradeTick
    from market.stream.ctp import CtpLiveDataFeed, CtpMdConfig

    symbol = _required_env("CTP_SYMBOL")
    exchange = os.getenv("CTP_EXCHANGE", "SHFE").upper()
    instrument_id = InstrumentId.from_str(f"{symbol}.{exchange}")
    timeout = float(os.getenv("CTP_MARKET_TIMEOUT", "30"))
    quote_received = threading.Event()
    trade_received = threading.Event()

    feed = CtpLiveDataFeed(
        CtpMdConfig(
            front=_required_env("CTP_MD_ADDRESS"),
            broker_id=_required_env("CTP_BROKER_ID"),
            user_id=_required_env("CTP_ACCOUNT_ID"),
            password=_required_env("CTP_PASSWORD"),
            flow_path=os.getenv("CTP_MD_FLOW_PATH", "/tmp/bomber-ctp-live-feed"),
            production_mode=_env_bool("CTP_PRODUCTION_MODE", False),
        ),
    )
    # InstrumentMeta 决定标准价格和数量的精度；配置错误会导致构造值失真。
    feed.register_instrument(
        InstrumentMeta(
            instrument_id=instrument_id,
            price_precision=int(os.getenv("CTP_PRICE_PRECISION", "0")),
            size_precision=int(os.getenv("CTP_SIZE_PRECISION", "0")),
            price_increment=Decimal(os.getenv("CTP_PRICE_INCREMENT", "1")),
            multiplier=Decimal(os.getenv("CTP_MULTIPLIER", "10")),
            currency=os.getenv("CTP_CURRENCY", "CNY"),
            exchange=exchange,
        ),
    )

    def on_quote(tick: QuoteTick) -> None:
        """只检查第一份Quote，避免持续行情产生大量日志。"""
        if quote_received.is_set():
            return
        logger.info(
            "标准 QuoteTick: %s bid=%s@%s ask=%s@%s "
            "price_precision=%s/%s size_precision=%s/%s ts=%s",
            tick.instrument_id,
            tick.bid_price,
            tick.bid_size,
            tick.ask_price,
            tick.ask_size,
            tick.bid_price.precision,
            tick.ask_price.precision,
            tick.bid_size.precision,
            tick.ask_size.precision,
            tick.ts_event,
        )
        if tick.bid_size.precision != tick.ask_size.precision:
            raise AssertionError("QuoteTick bid/ask size precision mismatch")
        quote_received.set()

    def on_trade(tick: TradeTick) -> None:
        """等待由累计Volume增加推导出的第一份有效TradeTick。"""
        if trade_received.is_set():
            return
        logger.info(
            "标准 TradeTick: %s price=%s size=%s trade_id=%s ts=%s",
            tick.instrument_id,
            tick.price,
            tick.size,
            tick.trade_id,
            tick.ts_event,
        )
        if tick.price.as_double() <= 0:
            raise AssertionError(f"TradeTick price must be positive, got {tick.price}")
        if tick.size.as_double() <= 0:
            raise AssertionError(f"TradeTick size must be positive, got {tick.size}")
        trade_received.set()

    feed.register_quote_tick_handler(on_quote)
    feed.register_trade_tick_handler(on_trade)
    # connect() 前先登记订阅，Feed 登录成功后会自动恢复这些订阅。
    feed.subscribe(instrument_id, DataType.QUOTE_TICK)
    feed.subscribe(instrument_id, DataType.TRADE_TICK)

    logger.info("启动正式 CTP Feed: %s", instrument_id)
    try:
        feed.connect()
        if not feed.wait_until_ready(float(os.getenv("CTP_CONNECT_TIMEOUT", "15"))):
            raise TimeoutError("CTP Feed 登录或订阅提交超时")
        # Quote通常每个快照都能生成；Trade只有累计Volume增加时才会生成。
        if not quote_received.wait(timeout):
            raise TimeoutError(f"{timeout:g} 秒内未收到标准 QuoteTick")
        if not trade_received.wait(timeout):
            raise TimeoutError(
                f"{timeout:g} 秒内未收到标准 TradeTick；"
                "可能是期间累计 Volume 没有增加",
            )
        print("CTP standard feed OK")
    finally:
        logger.info("停止正式 CTP Feed")
        feed.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTP行情链路分层探针；不连接交易前置")
    parser.add_argument("--stage", choices=("1", "2", "3", "4", "5"), default="5")
    args = parser.parse_args()
    {"1": test1, "2": test2, "3": test3, "4": test4, "5": test5}[args.stage]()
