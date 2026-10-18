"""Control-thread preparation while waiting; readiness checks only read cached facts."""
import math
import time


def wait_for_session_ready(session, timeout_seconds, *, lost=None, monotonic=None, sleep=None):
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("就绪等待须为有限正数")
    clock, pause = monotonic or time.monotonic, sleep or time.sleep
    deadline = clock() + timeout_seconds
    while True:
        if getattr(session.runner, "failure", None):
            raise RuntimeError(session.runner.failure)
        if lost is not None and lost():
            raise RuntimeError("等待就绪期间通道连接丢失")
        if clock() >= deadline:
            result = getattr(getattr(session, "readiness", None), "last_result", None)
            detail = result.as_dict() if result is not None else None
            raise RuntimeError(f"{timeout_seconds:g}秒内行情/执行未就绪: {detail}")
        # Always refresh before session_ready(); the old snapshot may have expired
        # during network setup. Database IO remains outside callback/frame locks.
        refresh = getattr(getattr(session, "references", None), "refresh", None)
        if callable(refresh):
            refresh()
        poll = getattr(session.runner, "poll_features", None)
        if callable(poll):
            poll()
        if lost is not None and lost():
            raise RuntimeError("等待就绪期间通道连接丢失")
        if clock() >= deadline:
            continue
        if session.runner.session_ready():
            return
        pause(.2)
