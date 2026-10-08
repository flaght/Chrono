"""管理在线积木的准备、运行、结束与资源释放；通道规则由控制器注入。"""

from contextlib import ExitStack
from dataclasses import dataclass
import time
from typing import Protocol

from bomber.framework.trader.contracts import RuntimeMode


@dataclass
class LiveRunContext:
    trading_day: str
    resources: ExitStack
    references: object = None


class LiveSessionControllerPort(Protocol):
    """通道实现会话规则；Runtime负责调度及释放输入工厂登记的资源。"""
    @property
    def event_label(self) -> str: ...
    def prepare(self, resources: ExitStack) -> LiveRunContext: ...
    def finish_preparation(self, context: LiveRunContext) -> None: ...
    def start(self, session, context: LiveRunContext) -> None: ...
    def poll(self, session) -> bool: ...
    def verify(self, session) -> None: ...
    def shutdown(self, session, context: LiveRunContext | None) -> dict: ...
    def snapshot(self, session, context: LiveRunContext | None) -> dict: ...
    def events(self, session): ...


class ManagedLiveRuntime:
    """提供run/start/stop，独立于策略及CTP；一次实例只运行一次会话。

    控制器提供prepare、finish_preparation、start、poll、verify、shutdown、
    snapshot及events。输入工厂与组装工厂依次运行；前者创建参考/行情积木，
    后者创建执行/策略/绑定。通道预检在后者安装持久化钩子前结束。
    """
    mode = RuntimeMode.LIVE

    def __init__(self, runtime_id, *, controller: LiveSessionControllerPort, prepare_inputs, assemble_session,
                 report, seconds, progress=None, monotonic=None, sleep=None):
        if not runtime_id.strip() or seconds <= 0:
            raise ValueError("运行标识不能为空，运行时长须为正数")
        self.runtime_id = runtime_id
        self.controller = controller
        self.prepare_inputs = prepare_inputs
        self.assemble_session = assemble_session
        self.report = report
        self.seconds = seconds
        self.progress = progress or (lambda session: None)
        self._monotonic = monotonic or (lambda: time.monotonic())
        self._sleep = sleep or (lambda seconds: time.sleep(seconds))
        self._resources = ExitStack()
        self.context = None
        self.inputs = None
        self.session = None
        self.result = None
        self._started = False
        self._preparation_started = False
        self._stopped = False
        self._status = "failed"
        self.failure = None

    def start(self):
        if self._stopped:
            raise RuntimeError("已结束的实盘Runtime不能重新启动")
        if self._started:
            return
        self.report.begin()
        try:
            self._preparation_started = True
            self.context = self.controller.prepare(self._resources)
            self.inputs = self.prepare_inputs(self.context)
            self.controller.finish_preparation(self.context)
            self.session = self.assemble_session(self.inputs, self.context)
            if self.session.runner.mode is not RuntimeMode.LIVE:
                raise ValueError("ManagedLiveRuntime仅接受LIVE Runner")
            self.controller.start(self.session, self.context)
            self._started = True
        except BaseException as error:
            self.failure = error
            self.stop()
            raise

    def run(self):
        if self.result is not None and self._status == "passed":
            return self.result
        try:
            self.start()
            deadline = self._monotonic() + self.seconds
            seen = 0
            while self._monotonic() < deadline:
                if not self.controller.poll(self.session):
                    break
                self.progress(self.session)
                rows = self.controller.events(self.session)
                for event in rows[seen:]:
                    print(f"{self.controller.event_label}: {event}", flush=True)
                seen = len(rows)
                self._sleep(0.2)
            self.controller.verify(self.session)
            self._status = "passed"
        except BaseException as error:
            self.failure = error
            raise
        finally:
            self.stop()
        return self.result

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        fields = {}
        errors = []
        try:
            if self._preparation_started:
                fields = self.controller.shutdown(self.session, self.context)
            errors.extend(fields.get("cleanup_errors", ()))
        except Exception as error:
            errors.append(str(error))
        finally:
            try:
                self._resources.close()
            except Exception as error:
                errors.append(str(error))
            self._started = False
        try:
            if self._preparation_started:
                fields.update(self.controller.snapshot(self.session, self.context))
        except Exception as error:
            errors.append(str(error))
        fields["cleanup_errors"] = errors
        if errors:
            self._status = "failed"
        if self.failure is not None:
            fields["failure"] = str(self.failure)
        if self.report.path is not None:
            self.result = self.report.write(status=self._status, fields=fields,
                session=self.session, events=self.controller.events(self.session))
        if errors:
            raise RuntimeError("停机查询／撤单未完成: " + "; ".join(errors))
