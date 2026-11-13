# -*- coding: utf-8 -*-
"""CTP 查询串行调度器。

所有账户共享一个发送节流器；每个请求收到 bIsLast 后才执行下一项。
"""

import threading
import time
from collections import deque


QUERY_METHODS = {
    "funds": "reqQryTradingAccount",
    "positions": "reqQryInvestorPosition",
    "orders": "reqQryOrder",
    "trades": "reqQryTrade",
    "position_details": "reqQryInvestorPositionDetail",
}


class QueryTask(object):

    def __init__(self, gateway_name, query_type, timeout=30.0,
                 ready_timeout=120.0):
        self.gateway_name = gateway_name
        self.query_type = query_type
        self.timeout = timeout
        self.ready_timeout = ready_timeout
        self.reqid = None
        self.rows = []
        self.error = None
        self.response_done = threading.Event()
        self.completed = threading.Event()


class QueryScheduler(object):

    def __init__(self, main_engine, account_configs, result_handler,
                 min_interval=1.1):
        self.main_engine = main_engine
        self.account_configs = account_configs
        self.result_handler = result_handler
        self.min_interval = float(min_interval)
        self._queue = deque()
        self._pending = {}
        self._current = None
        self._condition = threading.Condition()
        self._stopped = threading.Event()
        self._thread = None
        self._last_send_time = 0.0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stopped.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="ctp-monitor-query",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stopped.set()
        with self._condition:
            if self._current:
                self._current.error = "查询调度器已停止"
                self._current.response_done.set()
            while self._queue:
                task = self._queue.popleft()
                task.error = "查询调度器已停止"
                task.completed.set()
            self._pending.clear()
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=5.0)

    def request(self, gateway_name, query_type, timeout=30.0,
                ready_timeout=120.0, deduplicate=True):
        if query_type not in QUERY_METHODS:
            raise ValueError("未知查询类型: {0}".format(query_type))
        if gateway_name not in self.account_configs:
            raise ValueError("未注册网关: {0}".format(gateway_name))

        key = (gateway_name, query_type)
        with self._condition:
            if deduplicate and key in self._pending:
                return self._pending[key]
            task = QueryTask(
                gateway_name, query_type, timeout, ready_timeout
            )
            self._pending[key] = task
            self._queue.append(task)
            self._condition.notify_all()
            return task

    def handle_response(self, payload):
        with self._condition:
            task = self._current
            if not task:
                return
            if payload.get("gateway_name") != task.gateway_name:
                return
            if payload.get("query_type") != task.query_type:
                return
            if payload.get("reqid") != task.reqid:
                return

            error = payload.get("error") or {}
            if error.get("ErrorID"):
                task.error = "CTP错误 {0}: {1}".format(
                    error.get("ErrorID"), error.get("ErrorMsg", "")
                )
            data = payload.get("data") or {}
            if data:
                task.rows.append(data)
            if payload.get("last"):
                task.response_done.set()

    def _run(self):
        while not self._stopped.is_set():
            with self._condition:
                while not self._queue and not self._stopped.is_set():
                    self._condition.wait(timeout=1.0)
                if self._stopped.is_set():
                    break
                task = self._queue.popleft()
                self._current = task

            try:
                self._execute(task)
                if not task.error:
                    self.result_handler(task)
            except Exception as exc:
                task.error = str(exc)
            finally:
                if task.error and not self._stopped.is_set():
                    print("[{0}] {1} 查询失败: {2}".format(
                        task.gateway_name, task.query_type, task.error
                    ))
                key = (task.gateway_name, task.query_type)
                with self._condition:
                    self._pending.pop(key, None)
                    self._current = None
                task.completed.set()

    def _wait_ready(self, task):
        deadline = time.time() + task.ready_timeout
        while not self._stopped.is_set() and time.time() < deadline:
            gateway = self.main_engine.get_gateway(task.gateway_name)
            td_api = getattr(gateway, "td_api", None) if gateway else None
            if td_api and getattr(td_api, "login_status", False):
                contract_ready = getattr(td_api, "contract_inited", True)
                if contract_ready:
                    return td_api
            if td_api and (
                getattr(td_api, "login_failed", False)
                or getattr(td_api, "auth_failed", False)
            ):
                raise RuntimeError(
                    "{0} CTP认证或登录失败".format(task.gateway_name)
                )
            time.sleep(0.2)
        raise TimeoutError(
            "等待 {0} 登录及初始化超时".format(task.gateway_name)
        )

    def _execute(self, task):
        td_api = self._wait_ready(task)
        method_name = QUERY_METHODS[task.query_type]
        method = getattr(td_api, method_name, None)
        if not method:
            raise RuntimeError(
                "当前 CTP API 不支持 {0}".format(method_name)
            )

        config = self.account_configs[task.gateway_name]
        request = {
            "BrokerID": config.get("broker_id", ""),
            "InvestorID": config.get("account_id", ""),
        }
        deadline = time.time() + task.timeout

        while not self._stopped.is_set():
            wait_seconds = self.min_interval - (
                time.time() - self._last_send_time
            )
            if wait_seconds > 0:
                self._stopped.wait(wait_seconds)
            if self._stopped.is_set():
                raise RuntimeError("查询调度器已停止")

            td_api.reqid += 1
            task.reqid = td_api.reqid
            result = method(request, task.reqid)
            self._last_send_time = time.time()
            if result == 0:
                break
            if time.time() + self.min_interval >= deadline:
                raise RuntimeError(
                    "{0} 发送失败，CTP返回码: {1}".format(
                        method_name, result
                    )
                )

        remaining = max(0.0, deadline - time.time())
        if not task.response_done.wait(remaining):
            raise TimeoutError(
                "{0} {1} 查询响应超时".format(
                    task.gateway_name, task.query_type
                )
            )
        if task.error:
            raise RuntimeError(task.error)

