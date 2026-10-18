from types import SimpleNamespace as NS
import unittest
from tests.test_live_common import load

waiting = load("_test_live_runtime.live.readiness", "bomber/framework/trader/runtime/live/readiness.py")


class ReadinessWaitTests(unittest.TestCase):
    def test_refresh_before_gate_keeps_long_startup_fresh(self):
        now, observed, polls = [0.0], [0.0], []
        def refresh(): observed[0] = now[0]
        def ready():
            if now[0]-observed[0] > 30:
                raise RuntimeError("参考过期")
            return now[0] >= 31
        session = NS(references=NS(refresh=refresh), runner=NS(failure=None, session_ready=ready,
            poll_features=lambda: polls.append(now[0])))
        waiting.wait_for_session_ready(session, 40, monotonic=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
        self.assertGreaterEqual(now[0], 31)
        self.assertTrue(polls)

    def test_no_market_times_out_with_cached_channel_reason(self):
        now = [0.0]
        result = NS(as_dict=lambda: {"issues": ["WAITING_FOR_FIRST_EVENT"]})
        session = NS(references=NS(refresh=lambda: None), runner=NS(session_ready=lambda: False, failure=None),
            readiness=NS(last_result=result))
        with self.assertRaisesRegex(RuntimeError, "WAITING_FOR_FIRST_EVENT"):
            waiting.wait_for_session_ready(session, 1, monotonic=lambda: now[0],
                sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))

    def test_refresh_failure_and_connection_loss_never_open_gate(self):
        calls = []
        def refresh(): raise RuntimeError("资料版本变化")
        session = NS(references=NS(refresh=refresh), runner=NS(failure=None, session_ready=lambda: calls.append("gate")))
        with self.assertRaisesRegex(RuntimeError, "版本变化"):
            waiting.wait_for_session_ready(session, 1)
        self.assertFalse(calls)
        with self.assertRaisesRegex(RuntimeError, "连接丢失"):
            waiting.wait_for_session_ready(session, 1, lost=lambda: True)


if __name__ == "__main__":
    unittest.main()
