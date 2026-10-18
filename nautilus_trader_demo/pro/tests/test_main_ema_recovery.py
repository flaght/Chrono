"""01持仓接管/单次恢复的原生实体装配回归；柜台与行情均为假API。"""

from contextlib import ExitStack, redirect_stderr
from decimal import Decimal
from importlib import import_module
from io import StringIO
from types import SimpleNamespace as NS
import unittest

fixtures = import_module('tests.run_main_ema_live')
live = import_module('demos.01_main_ema.run_live')


class MainEmaRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.LiveTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def resume_session(self, original, *, api=fixtures.AdoptTdApi, replay=False, save=True):
        if save:
            original.manager.save()
        original.runner.stop()
        original.driver.stop()
        if hasattr(original, 'adoption_resources'):
            original.adoption_resources.close()
        transport, driver, _ = self.fixture.transport_driver(orders=True, api_base=api)
        if replay:
            transport.front = 'tcp://182.254.243.31:40001'
            transport.production_mode = True
        upstream = fixtures.ManualMd()
        if replay: upstream.latest_trading_day = '20260921'
        restored = live.assemble(original.args, self.fixture.reference(), driver, upstream)
        restored.transport, restored.args = transport, original.args
        controller = live.MainEmaSessionLifecycle(transport,
            md_front='tcp://182.254.243.31:40011' if replay else 'tcp://fake:1234', orders=True,
            environment='replay' if replay else 'realtime',
            replay_md_trading_day='20260921' if replay else None,
            expected_trading_day='20260922', resume=True)
        controller.driver = driver
        resources = ExitStack()
        self.addCleanup(resources.close)
        context = controller.prepare(resources)
        context.references = restored.references
        controller.finish_preparation(context)
        try:
            controller.start(restored, context)
        except Exception:
            controller.shutdown(restored, context)
            raise
        self.addCleanup(restored.runner.stop)
        restored.controller = controller
        return restored

    def test_replay_adopted_short_is_owned_and_not_opened_again(self):
        s = self.fixture.session(orders=True, replay=True, adopt=True)
        self.assertEqual(s.strategy.position('rb_main'), -1)
        for minute in range(5): self.fixture.tick(s, minute, 3100 - minute * 10)
        self.assertEqual(s.strategy.last_target, -1)
        self.assertEqual(s.transport._api.sent, [])
        s.controller.verify(s)
        self.assertEqual(s.runner.position_manager.unassigned_position(live.CLIENT_ID, s.references.instrument_id), 0)

    def test_replay_adoption_uses_td_today_bucket_to_close_before_opening(self):
        s = self.fixture.session(orders=True, replay=True, adopt=True)
        for minute in range(4): self.fixture.tick(s, minute, 3100)
        self.assertEqual(len(s.transport._api.sent), 1)
        self.assertEqual(s.transport._api.sent[0]['CombOffsetFlag'], '3')
        self.fixture.fill(s, 0, duplicate=True)
        s.transport._api.gross = {}
        self.fixture.tick(s, 4, 3100)
        self.assertEqual(s.transport._api.sent[1]['CombOffsetFlag'], '0')
        self.assertEqual(s.strategy.fills_received, 1)

    def test_replay_resume_keeps_owned_short_rewarms_and_advances_revision(self):
        original = self.fixture.session(orders=True, replay=True, adopt=True)
        for minute in range(5): self.fixture.tick(original, minute, 3100 - minute * 10)
        revision = original.strategy._revision
        restored = self.resume_session(original, replay=True)
        self.assertTrue(restored.controller.resume_ready)
        self.assertEqual(restored.strategy.position('rb_main'), -1)
        self.assertEqual(restored.strategy.bars_used, 0)
        self.assertIsNone(restored.strategy.last_target)
        self.assertEqual(restored.strategy._revision, revision)
        for minute in range(5, 9): self.fixture.tick(restored, minute, 3100 - minute * 10)
        self.assertEqual(restored.strategy.bars_used, 3)
        self.assertEqual(restored.strategy._revision, revision + 1)
        self.assertEqual(restored.transport._api.sent, [])
        restored.controller.verify(restored)

    def test_restore_before_first_signal_still_preserves_adopted_position(self):
        original = self.fixture.session(orders=True, replay=True, adopt=True)
        restored = self.resume_session(original, replay=True)
        self.assertEqual(restored.strategy.position('rb_main'), -1)
        self.assertEqual(restored.strategy._revision, 0)
        self.assertFalse(restored.transport._api.sent)
        with self.assertRaisesRegex(RuntimeError, '尚未取得有效新EMA目标'):
            restored.controller.verify(restored)

    def test_resume_rejects_changed_profile_and_preserves_original_file(self):
        original = self.fixture.session(orders=True, adopt=True)
        original.manager.save()
        before = original.manager.repository.path.read_bytes()
        original.runner.stop()
        original.driver.stop()
        original.adoption_resources.close()
        transport, driver, _ = self.fixture.transport_driver(orders=True, api_base=fixtures.AdoptTdApi)
        args = NS(**{**vars(original.args), 'simnow_environment': 'replay', 'replay_md_trading_day': '20260921'})
        restored = live.assemble(args, self.fixture.reference(), driver, fixtures.ManualMd())
        with self.assertRaisesRegex(ValueError, '恢复会话身份'):
            restored.manager.restore()
        controller = live.MainEmaSessionLifecycle(transport, md_front='tcp://fake:1234',
            orders=True, resume=True, expected_trading_day='20260922')
        controller.driver = driver
        result = controller.shutdown(restored, None)
        self.assertTrue(result['resume_state_preserved'])
        self.assertEqual(original.manager.repository.path.read_bytes(), before)

    def test_resume_rejects_cabinet_position_mismatch_without_overwriting(self):
        original = self.fixture.session(orders=True, replay=True, adopt=True)
        original.manager.save()
        before = original.manager.repository.path.read_bytes()
        with self.assertRaisesRegex(RuntimeError, '柜台净仓|归属'):
            self.resume_session(original, api=fixtures.FlatTdApi, replay=True, save=False)
        self.assertEqual(original.manager.repository.path.read_bytes(), before)

    def test_resume_does_not_continue_saved_incomplete_target_before_new_ema(self):
        original = self.fixture.session(orders=True)
        # 下一分钟的首笔Tick才关闭上一分钟；minute=3先种下跌价，
        # minute=4关闭该Bar后才能产生反向目标和平多订单。
        for minute in range(4):
            self.fixture.tick(original, minute, 3100 if minute < 3 else 3000)
        self.assertEqual(len(original.transport._api.sent), 1)
        self.fixture.fill(original, 0)
        self.fixture.tick(original, 4, 3000)
        self.assertEqual(len(original.transport._api.sent), 2)
        self.assertEqual(original.transport._api.sent[1]['CombOffsetFlag'], '3')
        self.fixture.fill(original, 1)
        original.transport._api.gross = {}
        self.assertEqual(original.strategy.last_target, -1)
        self.assertEqual(original.strategy.position('rb_main'), 0)
        restored = self.resume_session(original, api=fixtures.FlatTdApi)
        for minute in range(5, 8): self.fixture.tick(restored, minute, 3200)
        self.assertEqual(restored.strategy.bars_used, 2)
        self.assertEqual(restored.transport._api.sent, [])
        self.fixture.tick(restored, 8, 3200)
        self.assertEqual(restored.strategy.last_target, 1)
        self.assertEqual(len(restored.transport._api.sent), 1)
        self.assertEqual(restored.transport._api.sent[0]['Direction'], '0')

    def test_single_resume_cli_requires_original_file_td_and_no_adoption(self):
        path = self.fixture.root / 'original.json'
        command = ['--connect', '--product', 'RB', '--mode', 'simnow', '--enable-orders',
            '--confirm-simnow', '--expected-source-day', '20260921', '--state-file', str(path),
            '--simnow-environment', 'replay', '--replay-md-trading-day', '20260921']
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            live.parse_args([*command, '--resume', '--expected-trading-day', '20260922'])
        path.write_text('{}')  # CLI只检查存在；内容校验由状态仓库负责。
        args = live.parse_args([*command, '--resume', '--expected-trading-day', '20260922'])
        self.assertTrue(args.resume)
        for flags in ([], ['--resume'], ['--resume', '--expected-trading-day', '20260922',
                '--adopt-existing-position', 'rb2704.SHFE', '--expected-position', '-1']):
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                live.parse_args([*command, *flags])

    def test_replay_adoption_cli_has_no_realtime_dependency(self):
        args = live.parse_args(['--connect', '--product', 'RB', '--mode', 'simnow',
            '--enable-orders', '--confirm-simnow', '--state-file', str(self.fixture.root / 'adopt.json'),
            '--expected-source-day', '20260921', '--expected-trading-day', '20260922',
            '--simnow-environment', 'replay', '--replay-md-trading-day', '20260921',
            '--adopt-existing-position', 'rb2704.SHFE', '--expected-position', '-1'])
        self.assertEqual(args.expected_position, -1)


if __name__ == '__main__':
    unittest.main()
