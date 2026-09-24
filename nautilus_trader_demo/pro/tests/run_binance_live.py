#!/usr/bin/env python3
"""币安实时行情在线订阅与 Cython C 实体分发执行脚本 (非单元测试)。"""

import json
import logging
import sys
import threading
import time
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
)
logger = logging.getLogger("BinanceRunner")

# 引入核心模型与订阅源
from market.basic.base import (
    DataType,
    InstrumentId,
    InstrumentMeta,
    QuoteTick,
    TradeTick,
)
from market.stream.bn import BNWSConfig, BNWSStreamDataFeed


# --------------------------------------------------------------------------------------
# 1. 业务端回调定义 (接收 Cython C 级紧凑数据实体)
# --------------------------------------------------------------------------------------

def on_trade_received(tick: TradeTick) -> None:
    """处理高频逐笔成交事件。"""
    print(
        f"[逐笔成交] 标的: {tick.instrument_id} | "
        f"价格: {tick.price.as_double():.2f} | "
        f"数量: {tick.size.as_double():.4f} | "
        f"TradeId: {tick.trade_id} | "
        f"事件时间: {tick.ts_event}"
    )


def on_quote_received(quote: QuoteTick) -> None:
    """处理最优盘口报价 (bookTicker) 事件。"""
    print(
        f"[盘口报价] 标的: {quote.instrument_id} | "
        f"买一: {quote.bid_price.as_double():.2f} (量: {quote.bid_size.as_double():.4f}) | "
        f"卖一: {quote.ask_price.as_double():.2f} (量: {quote.ask_size.as_double():.4f}) | "
        f"价差: {(quote.ask_price.as_double() - quote.bid_price.as_double()):.2f}"
    )


# --------------------------------------------------------------------------------------
# 2. 真实网络客户端封装 (基于 websocket-client 长连接)
# --------------------------------------------------------------------------------------

class RealBinanceLiveFeed(BNWSStreamDataFeed):
    """扩展 BNWSStreamDataFeed，增加真实 WebSocket 线程客户端。"""

    def __init__(self, config: BNWSConfig | None = None) -> None:
        super().__init__(config=config)
        self._ws_app = None
        self._net_thread: threading.Thread | None = None

    def _start_network_client(self) -> None:
        """启动后台线程维持 WebSocket 长连接。若未安装依赖则回退至模拟数据流。"""
        try:
            import websocket
        except ImportError:
            logger.warning(
                "未检测到 'websocket-client' 模块（可执行 `pip install websocket-client` 安装以连接币安实盘）。\n"
                "==> 正在启动内置实时行情发生器，为您演示完整的实体构造与回调分发流程..."
            )
            self._stop_sim = threading.Event()
            def _sim_loop():
                import random
                price = 65420.50
                trade_id = 100001
                while not self._stop_sim.is_set():
                    time.sleep(0.3)
                    delta = random.choice([-0.5, 0.0, 0.5, 1.0, -1.0])
                    price = round(price + delta, 2)
                    ts_now = int(time.time() * 1000)
                    
                    # 模拟 trade 报文
                    trade_msg = {
                        "e": "trade",
                        "s": "BTCUSDT",
                        "t": trade_id,
                        "p": f"{price:.2f}",
                        "q": f"{random.uniform(0.01, 1.5):.4f}",
                        "T": ts_now,
                    }
                    trade_id += 1
                    self.on_ws_message(trade_msg)

                    # 模拟 bookTicker 报文
                    book_msg = {
                        "s": "BTCUSDT",
                        "b": f"{price - 0.1:.2f}",
                        "B": f"{random.uniform(1.0, 5.0):.4f}",
                        "a": f"{price + 0.1:.2f}",
                        "A": f"{random.uniform(1.0, 5.0):.4f}",
                    }
                    self.on_ws_message(book_msg)

            self._net_thread = threading.Thread(
                target=_sim_loop,
                name="Binance-SimThread",
                daemon=True,
            )
            self._net_thread.start()
            return

        # 构造组合流订阅 URL (例如: btcusdt@trade / btcusdt@bookTicker)
        streams = list(self._active_streams)
        if not streams:
            # 默认订阅 btcusdt 逐笔和盘口
            stream_path = "btcusdt@trade/btcusdt@bookTicker"
        else:
            stream_path = "/".join(streams)

        ws_url = f"{self.config.ws_base_url}/stream?streams={stream_path}"
        logger.info(f"正在连接币安 WebSocket 流: {ws_url}")

        def on_message(ws, msg):
            try:
                data = json.loads(msg)
                # 币安组合流报文格式为 {"stream": "...", "data": {...}}
                payload = data.get("data", data)
                self.on_ws_message(payload)
            except Exception as e:
                logger.error(f"解析报文异常: {e}")

        def on_error(ws, err):
            logger.error(f"WebSocket 异常: {err}")

        def on_close(ws, close_status, close_msg):
            logger.info("WebSocket 连接已关闭。")

        self._ws_app = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        self._net_thread = threading.Thread(
            target=self._ws_app.run_forever,
            name="Binance-NetThread",
            daemon=True,
        )
        self._net_thread.start()

    def _stop_network_client(self) -> None:
        """安全断开 WebSocket 连接。"""
        if hasattr(self, "_stop_sim"):
            self._stop_sim.set()
        if self._ws_app:
            self._ws_app.close()
        if self._net_thread and self._net_thread.is_alive():
            self._net_thread.join(timeout=2.0)


# --------------------------------------------------------------------------------------
# 3. 主程序入口与运行循环
# --------------------------------------------------------------------------------------

def main() -> None:
    # A. 实例化币安流配置
    config = BNWSConfig(
        ws_base_url="wss://stream.binance.com:9443",
    )
    feed = RealBinanceLiveFeed(config=config)

    # B. 注册合约元数据（价格 2 位小数，数量 4 位小数）
    btc_inst = InstrumentId.from_str("BTCUSDT.BINANCE")
    meta = InstrumentMeta(
        instrument_id=btc_inst,
        price_precision=2,
        size_precision=4,
        exchange="BINANCE",
        currency="USDT",
    )
    feed.register_instrument(meta)

    # C. 注册事件处理监听器
    feed.register_trade_tick_handler(on_trade_received)
    feed.register_quote_tick_handler(on_quote_received)

    # D. 声明式订阅行情
    feed.subscribe(btc_inst, DataType.TRADE_TICK)
    feed.subscribe(btc_inst, DataType.QUOTE_TICK)

    # E. 建立连接并启动分派工作循环
    feed.connect()
    logger.info("行情订阅引擎启动成功，开始监听实时推送...")

    # F. 主线程维持运行 30 秒，演示实时接收
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        logger.info("用户主动中断...")
    finally:
        # G. 优雅断开数据源
        feed.disconnect()
        logger.info("已安全断开并退出。")


if __name__ == "__main__":
    main()
