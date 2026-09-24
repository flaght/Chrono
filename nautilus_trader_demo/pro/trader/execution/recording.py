"""不会向交易所发送订单的记录型执行客户端。"""

from __future__ import annotations

import threading

from trader.contracts import ExecutionRequest


class RecordingExecutionClient:
    """保存Runner产生的账户目标请求，用于在线探针和装配验收。

    本客户端刻意不实现撮合、仓位更新或网络调用。它证明策略已经完成
    ``行情 → 目标 → 组合净额 → 执行路由``，但不能被解释为订单已成交。
    """

    def __init__(self, client_id: str) -> None:
        if not client_id.strip():
            raise ValueError("client_id不能为空")
        self.client_id = client_id
        self._requests: list[ExecutionRequest] = []
        self._started = False
        self._lock = threading.RLock()

    @property
    def is_started(self) -> bool:
        with self._lock:
            return self._started

    @property
    def requests(self) -> tuple[ExecutionRequest, ...]:
        with self._lock:
            return tuple(self._requests)

    def start(self) -> None:
        with self._lock:
            self._started = True

    def stop(self) -> None:
        with self._lock:
            self._started = False

    def submit_targets(self, request: ExecutionRequest) -> None:
        with self._lock:
            if not self._started:
                raise RuntimeError("Recording执行客户端尚未启动")
            if request.client_id != self.client_id:
                raise ValueError("ExecutionRequest客户端不匹配")
            self._requests.append(request)

    def cancel_strategy(self, strategy_id: str) -> None:
        # 没有真实活动订单；保留接口以满足ExecutionClientPort。
        del strategy_id
