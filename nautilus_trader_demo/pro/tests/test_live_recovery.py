"""公共持仓恢复无网络回归：实际恢复代码配可控账户/Runner替身。"""

import ast
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from tests.test_live_common import ROOT, load

recovery = import_module('_test_live_runtime.live.recovery')
ctp = import_module('_test_live_runtime.live.channels.ctp_recovery')


@dataclass(frozen=True)
class Key:
    client_id: str
    instrument_id: str


class Positions:
    def __init__(self, quantity=-1):
        self.owned = {('ema', 'main'): Decimal(quantity)}
        self.account = {Key('td', 'RB.SHFE'): Decimal(quantity)}
        self.working = {}
        self.unassigned = Decimal(0)

    def snapshot(self):
        return NS(strategy_positions=self.owned, account_positions=self.account,
                  working_quantities=self.working)

    def position(self, sid, key):
        return self.owned.get((sid, key), Decimal(0))

    def account_position(self, cid, instrument):
        return self.account.get(Key(cid, instrument), Decimal(0))

    def unassigned_position(self, *args):
        return self.unassigned

    def working_quantity(self, cid, instrument):
        return self.working.get(Key(cid, instrument), Decimal(0))


class BaseLifecycle:
    def __init__(self, *, orders=True, adopt_instrument=None, expected_trading_day='20261009'):
        self.orders, self.adopt_instrument = orders, adopt_instrument
        self.expected_trading_day = expected_trading_day
        self.lock_namespace, self.legacy_lock_namespaces = 'test', ()
        self.ready_seconds, self.reconcile_seconds = 30, 10
        self.lost = []
        self.calls = []
        self.driver = NS(trading_day='20261009', start=lambda handler: self.calls.append('td.start'),
            stop=lambda: self.calls.append('td.stop'), reconcile_account_state=lambda: None,
            reconcile_active_orders=lambda: NS(orders=()))
        self.transport = NS(account_id='fake-account')

    def prepare(self, resources):
        self.calls.append('fresh.prepare')
        return NS(trading_day='20261009')

    def start(self, session, context):
        self.calls.append('fresh.start')

    def shutdown(self, session, context):
        self.calls.append('normal.shutdown')
        return {'cleanup_errors': []}

    def snapshot(self, session, context):
        return {}


class Lifecycle(ctp.CtpResumeLifecycle, BaseLifecycle):
    def validate_restored_session(self, session, day):
        self.calls.append('validate.local')
        ctp.validate_ctp_restored_positions(session, day, session.recovery_binding)
        return recovery.restored_target_revision(session, session.recovery_binding)


def session(controller):
    calls = controller.calls
    positions = Positions()
    snap = NS(long_total=Decimal(0), short_total=Decimal(1), net_position=Decimal(-1),
        long_today=Decimal(0), long_yesterday=Decimal(0),
        short_today=Decimal(1), short_yesterday=Decimal(0))
    target = NS(targets={'main': Decimal(-1)}, revision=7)
    binding = recovery.StrategyPositionBinding('ema', 'td', {'main': 'RB.SHFE'})
    runner = NS(position_manager=positions, accept_bars=False, fixed_spec='spec',
        target_store=NS(all=lambda: {'ema': target}),
        portfolio_coordinator=NS(state=lambda: NS(strategy_revisions={'ema': 7},
            contributions={'ema': {Key('td', 'RB.SHFE'): Decimal(-1)}})),
        start=lambda: calls.append('runner.start'), stop=lambda: calls.append('runner.stop'),
        session_ready=lambda: True, begin_bars=lambda: calls.append('begin.bars'))
    client = NS(client_id='td', start=lambda: calls.append('account.reconcile'),
        arm_demo=lambda token: calls.append('arm'), DEMO_CONFIRMATION='fake-token',
        order_state_machine=NS(states=lambda: ()),
        account_state=NS(balances={'CNY': NS(available=Decimal(100000))}))
    ledger = NS(state=lambda: NS(trading_day='20261009', positions={'RB.SHFE': snap}),
        snapshot=lambda instrument: snap)
    manager = NS(restore=lambda: (calls.append('restore') or NS(generation=9)),
        save=lambda: calls.append('save'))
    controller.transport.query_gross_positions = lambda: {'RB.SHFE': (Decimal(0), Decimal(1))}
    rows = [{'InstrumentID': 'RB', 'ExchangeID': 'SHFE', 'Position': 1,
             'HedgeFlag': '1', 'PosiDirection': '3', 'PositionDate': '1'}]
    controller.transport.query_position_details = lambda: rows
    return NS(runner=runner, client=client, ledger=ledger, manager=manager,
        driver=controller.driver, recovery_binding=binding, rows=rows, target=target,
        references=NS(spec='spec', refresh=lambda: calls.append('reference.refresh')))


class RecoveryTests(unittest.TestCase):
    def test_cross_section_failed_resume_uses_public_state_file_protection(self):
        aliases = {'bomber.framework.trader.runtime.live.recovery': recovery,
                   'bomber.framework.trader.runtime.live.channels.ctp_recovery': ctp}
        with patch.dict(sys.modules, aliases):
            hooks = load('_test_cross_section_recovery_hooks', 'demos/04_cross_section/recovery.py')
        self.assertTrue(issubclass(hooks.ResumeLifecycle, ctp.CtpResumeLifecycle))
        class Controller(hooks.ResumeLifecycle, BaseLifecycle):
            pass
        c = Controller(resume=True)
        s = session(c)
        result = c.shutdown(s, None)
        self.assertTrue(result['resume_state_preserved'])
        self.assertNotIn('save', c.calls)
        self.assertNotIn('normal.shutdown', c.calls)

    def scoped_controller(self, *, scoped=True):
        channel = import_module('_test_live_runtime.live.channels.ctp')
        calls = []
        gross = {'RB.SHFE': (Decimal(0), Decimal(1))}
        transport = NS(broker_id='9999', front='tcp://fake:1234', production_mode=True,
                       client_id='td', account_id='fake-account', query_gross_positions=lambda: dict(gross))
        sdk = NS(CtpNativeTraderDriver=lambda *args, **kwargs: NS())
        with patch.dict(sys.modules, {'bomber.framework.trader.execution.ctp': sdk}):
            c = channel.CtpSessionLifecycle(transport, md_front='tcp://fake:1234', orders=True,
                expected_trading_day='20261009', scope_to_reference_instruments=scoped)
        c.driver = NS(trading_day='20261009', submitted_orders=0,
            start=lambda callback: calls.append('td.start'), stop=lambda: calls.append('td.stop'),
            reconcile=lambda: {'RB.SHFE': Decimal(-1)},
            reconcile_account_state=lambda: NS(balances={'CNY': NS(available=Decimal(100000))}),
            reconcile_active_orders=lambda: NS(orders=()))
        refs = NS(instrument_ids={'AU': 'AU.SHFE', 'AG': 'AG.SHFE', 'SC': 'SC.INE'},
                  snapshot=lambda now: None, refresh_on_event=True)
        s = NS(position_scope=recovery.InstrumentPositionScope(refs.instrument_ids.values()),
            references=refs, upstream=NS(latest_trading_day='20261009'),
            runner=NS(accept_bars=False, failure=None, start=lambda: calls.append('runner.start'),
                stop=lambda: calls.append('runner.stop'), session_ready=lambda: True,
                reference_is_current=lambda: True, begin_bars=lambda: calls.append('begin.bars')),
            client=NS(start=lambda: calls.append('client.start'), refresh_account_state=lambda: None,
                arm_demo=lambda token: calls.append('arm'), DEMO_CONFIRMATION='fake', report_errors=(),
                set_risk_mode=lambda *args, **kwargs: calls.append('cancel.own')),
            manager=NS(save=lambda: calls.append('save')))
        c._expected_gross = lambda session: {}
        return channel, c, s, gross, calls

    def start_scoped_controller(self):
        channel, c, s, gross, calls = self.scoped_controller()
        with patch.object(channel, 'account_lock', return_value=nullcontext()), ExitStack() as resources:
            context = c.prepare(resources)
            context.references = s.references
            c.finish_preparation(context)
            c.start(s, context)
        return c, s, context, gross, calls

    def test_scope_copies_ids_and_preserves_mapping_keys(self):
        ids = ['AU.SHFE', 'SC.INE']
        scope = recovery.InstrumentPositionScope(ids)
        ids.append('RB.SHFE')
        self.assertFalse(scope.contains('RB.SHFE'))
        positions = {'AU.SHFE': 1, 'RB.SHFE': -1}
        self.assertEqual(scope.select(positions), {'AU.SHFE': 1})
        self.assertEqual(scope.external(positions), {'RB.SHFE': -1})
        self.assertEqual(positions, {'AU.SHFE': 1, 'RB.SHFE': -1})
        for invalid in ([], [None], [''], 'AU.SHFE'):
            with self.assertRaises(ValueError): recovery.InstrumentPositionScope(invalid)

    def test_scoped_fresh_start_ignores_external_position_without_adopting_it(self):
        c, s, context, gross, calls = self.start_scoped_controller()
        self.assertEqual(c.external_gross, gross)
        self.assertIsNone(c.adopted_position)
        self.assertLess(calls.index('save'), calls.index('arm'))
        self.assertIn('begin.bars', calls)

    def test_default_account_policy_still_requires_all_positions_flat(self):
        channel, c, s, gross, calls = self.scoped_controller(scoped=False)
        with patch.object(channel, 'account_lock', return_value=nullcontext()), ExitStack() as resources:
            with self.assertRaisesRegex(RuntimeError, '账户已有总仓'):
                c.prepare(resources)
        self.assertNotIn('arm', calls)

    def test_scope_must_be_bound_before_start_and_match_assembled_routes(self):
        channel, c, s, gross, calls = self.scoped_controller()
        with self.assertRaisesRegex(RuntimeError, '持仓范围'):
            c.start(s, NS(trading_day='20261009'))
        c.position_scope = recovery.InstrumentPositionScope(['AU.SHFE'])
        with self.assertRaisesRegex(RuntimeError, '持仓范围'):
            c.start(s, NS(trading_day='20261009'))
        self.assertNotIn('client.start', calls)

    def test_managed_unknown_gross_even_net_zero_blocks_preparation(self):
        channel, c, s, gross, calls = self.scoped_controller()
        gross['AU.SHFE'] = (Decimal(1), Decimal(1))
        with patch.object(channel, 'account_lock', return_value=nullcontext()), ExitStack() as resources:
            context = c.prepare(resources)
            context.references = s.references
            with self.assertRaisesRegex(RuntimeError, '账户已有总仓'):
                c.finish_preparation(context)
        self.assertNotIn('arm', calls)

    def test_scope_rechecks_managed_positions_after_preflight_relogin(self):
        channel, c, s, gross, calls = self.scoped_controller()
        with patch.object(channel, 'account_lock', return_value=nullcontext()), ExitStack() as resources:
            context = c.prepare(resources)
            context.references = s.references
            c.finish_preparation(context)
            gross['AU.SHFE'] = (Decimal(1), Decimal(0))
            with self.assertRaisesRegex(RuntimeError, '账户已有总仓'):
                c.start(s, context)
        self.assertNotIn('arm', calls)

    def test_scoped_poll_ignores_external_changes_but_rejects_managed_mismatch(self):
        c, s, context, gross, calls = self.start_scoped_controller()
        gross['RB.SHFE'] = (Decimal(0), Decimal(2))
        c._next_check = 0
        self.assertTrue(c.poll(s))
        self.assertEqual(c.external_gross['RB.SHFE'], (0, 2))
        gross['SC.INE'] = (Decimal(1), Decimal(0))
        c._next_check = 0
        with self.assertRaisesRegex(RuntimeError, '柜台总仓与本地账本不一致'):
            c.poll(s)

    def test_scoped_shutdown_preserves_and_reports_external_gross(self):
        c, s, context, gross, calls = self.start_scoped_controller()
        result = c.shutdown(s, context)
        self.assertEqual(result['cleanup_errors'], [])
        self.assertEqual(result['final_gross'], gross)
        self.assertEqual(result['final_managed_gross'], {})
        self.assertEqual(result['external_gross'], gross)
        self.assertEqual(gross, {'RB.SHFE': (0, 1)})

    def test_resume_allows_external_unowned_account_facts_and_detail_rows(self):
        c = Lifecycle(resume=True)
        s = session(c)
        s.position_scope = recovery.InstrumentPositionScope(s.recovery_binding.targets.values())
        s.runner.position_manager.account[Key('td', 'AU.SHFE')] = Decimal(-1)
        c.transport.query_gross_positions = lambda: {'RB.SHFE': (0, 1), 'AU.SHFE': (0, 1)}
        s.rows.append({'InstrumentID': 'AU', 'ExchangeID': 'SHFE', 'Position': 1,
                       'PosiDirection': '3', 'PositionDate': '1', 'HedgeFlag': '1'})
        c.start(s, NS(trading_day='20261009'))
        self.assertTrue(c.resume_ready)
        self.assertEqual(s.runner.position_manager.owned, {('ema', 'main'): Decimal(-1)})
        self.assertEqual(s.runner.position_manager.account[Key('td', 'AU.SHFE')], -1)

    def test_scoped_restore_still_rejects_foreign_owner_account_or_route(self):
        for fault in ('owner', 'client', 'route', 'unassigned'):
            c = Lifecycle(resume=True)
            s = session(c)
            s.position_scope = recovery.InstrumentPositionScope(['RB.SHFE'])
            if fault == 'owner': s.runner.position_manager.owned[('other', 'main')] = Decimal(1)
            if fault == 'client': s.runner.position_manager.account[Key('other', 'AU.SHFE')] = Decimal(-1)
            if fault == 'route': s.position_scope = recovery.InstrumentPositionScope(['AU.SHFE'])
            if fault == 'unassigned': s.runner.position_manager.unassigned = Decimal(1)
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                c.start(s, NS(trading_day='20261009'))
            self.assertNotIn('arm', c.calls)

    def test_scope_backend_refuses_external_order_before_driver_send(self):
        # 执行真实Backend方法，只替换本机无法加载的原生基类。
        tree = ast.parse((ROOT / 'bomber/framework/trader/execution/builders.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_CtpPortfolioBackend')
        sent = []
        class Base:
            def submit_order(self, order): sent.append(order.instrument_id)
        namespace = {'NautilusLiveExecutionBackend': Base}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-ctp-backend>', 'exec'), namespace)
        backend = namespace['_CtpPortfolioBackend']()
        backend.instrument_scope = recovery.InstrumentPositionScope(['AU.SHFE'])
        backend.submission_check = lambda: True
        with self.assertRaisesRegex(RuntimeError, '执行范围'):
            backend.submit_order(NS(instrument_id='RB.SHFE'))
        self.assertEqual(sent, [])
        backend.submit_order(NS(instrument_id='AU.SHFE'))
        self.assertEqual(sent, ['AU.SHFE'])

    def test_ctp_replay_adoption_is_independent_of_market_profile(self):
        module = import_module('_test_live_runtime.live.channels.ctp')
        sdk = NS(CtpNativeTraderDriver=lambda *args, **kwargs: NS())
        transport = NS(broker_id='9999', front='tcp://182.254.243.31:40001',
            production_mode=True, client_id='td', account_id='demo')
        with patch.dict(sys.modules, {'bomber.framework.trader.execution.ctp': sdk}):
            lifecycle = module.CtpSessionLifecycle(transport,
                md_front='tcp://182.254.243.31:40011', environment='replay',
                replay_md_trading_day='20261009', orders=True,
                adopt_instrument='RB.SHFE', expected_position=Decimal(-1))
        self.assertEqual(lifecycle.profile.bar_time_basis, 'receive')
        self.assertEqual(lifecycle.expected_position, -1)

    def ema_hooks(self):
        aliases = {'bomber.framework.trader.runtime.live.recovery': recovery,
                   'bomber.framework.trader.runtime.live.channels.ctp_recovery': ctp}
        with patch.dict(sys.modules, aliases):
            return load('_test_main_ema_recovery_hooks', 'demos/01_main_ema/recovery.py')

    def test_ema_restore_sets_revision_without_bound_strategy_or_old_signal(self):
        hooks = self.ema_hooks()
        s = session(Lifecycle(resume=True))
        s.strategy = NS(config=NS(target_key='main', quantity=Decimal(1)),
                        _revision=0, _revision_lock=threading.Lock())
        held = []
        s.runner.hold_restored_target = lambda *args: held.append(args)
        self.assertEqual(hooks.validate_restored_session(s, '20261009'), 7)
        self.assertEqual(s.strategy._revision, 7)
        self.assertEqual(held, [('ema', 7)])
        s.runner.position_manager.owned[('ema', 'main')] = Decimal(-2)
        s.runner.position_manager.account[Key('td', 'RB.SHFE')] = Decimal(-2)
        with self.assertRaisesRegex(RuntimeError, '超过本策略'):
            hooks.validate_restored_session(s, '20261009')

    def test_ema_verify_matching_adopted_position_needs_no_new_fill(self):
        hooks = self.ema_hooks()
        c = Lifecycle(resume=True)
        s = session(c)
        s.strategy = NS(config=NS(slow_period=3, target_key='main'), bars_used=3,
            last_target=Decimal(-1), position=lambda key: Decimal(-1))
        s.runner.failure = None
        s.client.report_errors = ()
        s.references.instrument_id = 'RB.SHFE'
        hooks.verify_managed_position(s, c.transport)
        s.strategy.bars_used = 2
        with self.assertRaisesRegex(RuntimeError, '尚未取得有效新EMA目标'):
            hooks.verify_managed_position(s, c.transport)
        s.strategy.bars_used, s.strategy.last_target = 3, Decimal(1)
        with self.assertRaisesRegex(RuntimeError, '最新EMA目标尚未完成'):
            hooks.verify_managed_position(s, c.transport)

    def test_binding_copies_routes_and_rejects_ambiguous_ownership(self):
        routes = {'main': 'RB.SHFE'}
        bound = recovery.StrategyPositionBinding('ema', 'td', routes)
        routes['main'] = 'AU.SHFE'
        self.assertEqual(bound.targets['main'], 'RB.SHFE')
        for routes in ({}, {'a': 'RB.SHFE', 'b': 'RB.SHFE'}, {'': 'RB.SHFE'}):
            with self.assertRaises(ValueError):
                recovery.StrategyPositionBinding('ema', 'td', routes)

    def test_identity_checkpoint_rejects_each_changed_session_field(self):
        identity = {'strategy': 'ema', 'environment': 'replay', 'md_day': '20261009',
                    'td_day': '20261009', 'fast': 3, 'instrument': 'RB.SHFE'}
        checkpoint = recovery.SessionIdentityCheckpoint(identity)
        saved = checkpoint.snapshot_state()
        checkpoint.restore_state(saved)
        for key in identity:
            changed = checkpoint.snapshot_state()
            changed['identity'][key] = 'changed'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, '恢复会话身份'):
                checkpoint.restore_state(changed)
        saved['identity']['instrument'] = 'AU.SHFE'
        self.assertEqual(checkpoint.identity['instrument'], 'RB.SHFE')

    def test_restoration_orders_checks_and_authorization_are_in_sequence(self):
        c = Lifecycle(resume=True)
        s = session(c)
        c.start(s, NS(trading_day='20261009'))
        self.assertEqual(c.calls, ['restore', 'validate.local', 'account.reconcile',
            'runner.start', 'reference.refresh', 'save', 'arm', 'begin.bars'])
        self.assertTrue(c.resume_ready)
        self.assertEqual(c.restored_generation, 9)

    def test_fresh_lifecycle_still_uses_base_without_restore(self):
        c = Lifecycle()
        c.prepare(None)
        c.start(None, None)
        self.assertEqual(c.calls, ['fresh.prepare', 'fresh.start'])

    def test_resume_and_adoption_or_recording_are_mutually_exclusive(self):
        for kwargs in ({'orders': False}, {'adopt_instrument': 'RB.SHFE'},
                       {'expected_trading_day': None}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Lifecycle(resume=True, **kwargs)

    def test_prepare_checks_td_and_unknown_orders_before_restore(self):
        for mismatch in ('day', 'orders'):
            c = Lifecycle(resume=True)
            if mismatch == 'day':
                c.driver.trading_day = '20261012'
            else:
                c.driver.reconcile_active_orders = lambda: NS(orders=('unknown',))
            with patch.object(ctp, 'account_lock', return_value=nullcontext()), ExitStack() as stack:
                with self.assertRaises(RuntimeError):
                    c.prepare(stack)
            self.assertFalse(c.resume_ready)
            self.assertNotIn('arm', c.calls)

    def test_restore_missing_file_never_becomes_a_new_session(self):
        c = Lifecycle(resume=True)
        s = session(c)
        s.manager.restore = lambda: None
        with self.assertRaisesRegex(RuntimeError, '文件不存在'):
            c.start(s, NS(trading_day='20261009'))
        self.assertNotIn('account.reconcile', c.calls)

    def test_each_account_mismatch_blocks_authorization(self):
        for fault in ('owned', 'unassigned', 'gross', 'bucket', 'ledger', 'orders', 'funds'):
            c = Lifecycle(resume=True)
            s = session(c)
            if fault == 'owned': s.runner.position_manager.owned[('ema', 'main')] = Decimal(0)
            if fault == 'unassigned': s.runner.position_manager.unassigned = Decimal(1)
            if fault == 'gross': c.transport.query_gross_positions = lambda: {'AU.SHFE': (1, 0)}
            if fault == 'bucket': s.rows[0]['PositionDate'] = '2'
            if fault == 'ledger': s.ledger.snapshot('RB.SHFE').net_position = Decimal(0)
            if fault == 'orders': c.driver.reconcile_active_orders = lambda: NS(orders=('unknown',))
            if fault == 'funds': s.client.account_state.balances['CNY'].available = Decimal(0)
            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                c.start(s, NS(trading_day='20261009'))
            self.assertNotIn('arm', c.calls)
            self.assertFalse(c.resume_ready)

    def test_foreign_strategy_and_foreign_account_are_rejected(self):
        for kind in ('strategy', 'account'):
            s = session(Lifecycle(resume=True))
            if kind == 'strategy': s.runner.position_manager.owned[('other', 'main')] = Decimal(0)
            else: s.runner.position_manager.account[Key('other-td', 'RB.SHFE')] = Decimal(0)
            with self.assertRaises(RuntimeError):
                ctp.validate_ctp_restored_positions(s, '20261009', s.recovery_binding)

    def test_changed_ledger_day_or_instrument_blocks_restore(self):
        for state in (NS(trading_day='20261012', positions={}),
                      NS(trading_day='20261009', positions={'AU.SHFE': None})):
            s = session(Lifecycle(resume=True))
            s.ledger.state = lambda: state
            with self.assertRaisesRegex(RuntimeError, '相同交易日'):
                ctp.validate_ctp_restored_positions(s, '20261009', s.recovery_binding)

    def test_unfinished_local_order_or_working_quantity_blocks_restore(self):
        for working in (True, False):
            s = session(Lifecycle(resume=True))
            if working: s.runner.position_manager.working[Key('td', 'RB.SHFE')] = Decimal(1)
            else: s.client.order_state_machine.states = lambda: (NS(status=NS(is_terminal=False)),)
            with self.assertRaisesRegex(RuntimeError, '活动订单已结束'):
                ctp.validate_ctp_restored_positions(s, '20261009', s.recovery_binding)

    def test_save_failure_never_authorizes_and_preserves_original_file(self):
        with TemporaryDirectory() as directory:
            original = Path(directory) / 'original.json'
            original.write_bytes(b'original-checkpoint')
            c = Lifecycle(resume=True)
            s = session(c)
            s.manager.save = lambda: (_ for _ in ()).throw(OSError('disk failed'))
            with self.assertRaises(OSError): c.start(s, NS(trading_day='20261009'))
            result = c.shutdown(s, None)
            self.assertTrue(result['resume_state_preserved'])
            self.assertNotIn('arm', c.calls)
            self.assertEqual(original.read_bytes(), b'original-checkpoint')

    def test_failed_restore_shutdown_stops_resources_without_save_or_cancel(self):
        c = Lifecycle(resume=True)
        s = session(c)
        result = c.shutdown(s, None)
        self.assertEqual(c.calls, ['runner.stop', 'td.stop'])
        self.assertTrue(result['resume_state_preserved'])

    def test_newly_changed_day_or_reference_blocks_before_authorization(self):
        for failure in ('day', 'reference', 'reference_version', 'health'):
            c = Lifecycle(resume=True)
            s = session(c)
            if failure == 'day': s.client.start = lambda: setattr(c.driver, 'trading_day', '20261012')
            if failure == 'reference': s.references.refresh = lambda: setattr(s.references, 'spec', 'other')
            if failure == 'reference_version': s.runner.reference_is_current = lambda: False
            if failure == 'health':
                states = iter((True, False))
                s.runner.session_ready = lambda: next(states)
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                c.start(s, NS(trading_day='20261009'))
            self.assertNotIn('arm', c.calls)

    def test_restored_target_keys_and_revision_must_match(self):
        s = session(Lifecycle(resume=True))
        self.assertEqual(recovery.restored_target_revision(s, s.recovery_binding), 7)
        s.target.targets = {'unknown': Decimal(1)}
        with self.assertRaisesRegex(RuntimeError, '目标键'):
            recovery.restored_target_revision(s, s.recovery_binding)
        s.target.targets = {'main': Decimal(1)}
        s.runner.portfolio_coordinator.state = lambda: NS(strategy_revisions={'ema': 8})
        with self.assertRaisesRegex(RuntimeError, '组合版本'):
            recovery.restored_target_revision(s, s.recovery_binding)
        for contribution in ({Key('other-td', 'RB.SHFE'): Decimal(1)},
                             {Key('td', 'AU.SHFE'): Decimal(1)},
                             {Key('td', 'RB.SHFE'): Decimal(-1)}):
            s.runner.portfolio_coordinator.state = lambda: NS(strategy_revisions={'ema': 7},
                contributions={'ema': contribution})
            with self.subTest(contribution=contribution), self.assertRaisesRegex(RuntimeError, '组合贡献'):
                recovery.restored_target_revision(s, s.recovery_binding)

    def test_checkpoint_before_first_signal_can_restore_without_inventing_target(self):
        s = session(Lifecycle(resume=True))
        s.runner.target_store.all = lambda: {}
        s.runner.portfolio_coordinator.state = lambda: NS(strategy_revisions={})
        self.assertEqual(recovery.restored_target_revision(s, s.recovery_binding, allow_empty=True), 0)
        with self.assertRaises(RuntimeError): recovery.restored_target_revision(s, s.recovery_binding)

    def test_common_modules_have_no_strategy_or_foreign_channel_imports(self):
        for name in ('recovery.py', 'channels/ctp_recovery.py'):
            path = ROOT / 'bomber/framework/trader/runtime/live' / name
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn('demos', node.module or '')
                    self.assertNotIn('binance', node.module or '')
        for demo, other in (('01_main_ema', '04_cross_section'), ('04_cross_section', '01_main_ema')):
            for name in ('run_live.py', 'recovery.py'):
                self.assertNotIn(f'demos.{other}', (ROOT / 'demos' / demo / name).read_text())

    def test_real_runner_holds_old_target_until_new_revision(self):
        # 运行真实的三个方法；Base只记录续执行，免去原生Bar实体依赖。
        path = ROOT / 'bomber/framework/trader/live_roles.py'
        tree = ast.parse(path.read_text())
        source = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FixedRoleLiveRunner')
        source.body = [n for n in source.body if isinstance(n, ast.FunctionDef)
                       and n.name in {'__init__', 'hold_restored_target', 'continue_execution_target'}]
        mode = object()
        class Base:
            def __init__(self, *args, **kwargs):
                self.mode, self._started, self.calls = mode, False, []
            def continue_execution_target(self, *args, **kwargs):
                self.calls.append(args)
                return True
        namespace = {'UnifiedStrategyRunner': Base, 'RuntimeMode': NS(LIVE=mode)}
        exec(compile(ast.Module(body=[source], type_ignores=[]), str(path), 'exec'), namespace)
        runner = namespace['FixedRoleLiveRunner']()
        runner._role_binding = NS(strategy=NS(strategy_id='ema'))
        target = NS(revision=7)
        runner.target_store = NS(get=lambda sid: target)
        runner.hold_restored_target('ema', 7)
        self.assertFalse(runner.continue_execution_target('ema', 'main', 1))
        self.assertEqual(runner.calls, [])
        target.revision = 8
        self.assertTrue(runner.continue_execution_target('ema', 'main', 2))
        self.assertEqual(len(runner.calls), 1)
        runner._started = True
        with self.assertRaises(ValueError): runner.hold_restored_target('ema', 8)


if __name__ == '__main__':
    unittest.main()
