"""文档选约规则及真实StrategyTemplate接口验证；不依赖原生撮合或数据库。"""
from argparse import Namespace
from dataclasses import replace
from datetime import date, datetime
from math import exp, sqrt
from pathlib import Path
from statistics import NormalDist
import sys
import json
from importlib import import_module
import tempfile
import types
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
signal_module = import_module('demos.09_option_delta.delta_signal')
Config, Option, years_to_expiry = signal_module.Config, signal_module.Option, signal_module.years_to_expiry
price, implied_greeks, select = signal_module.price, signal_module.implied_greeks, signal_module.select
entry = import_module('demos.09_option_delta.run_backtest')
load_options = entry.load_options
from bomber.framework.dataprep.paths import resolve_option_args
from bomber.framework.dataprep.scenarios.option_chain import read_research_frame as read_frame


def resolve_paths(args, *, validate=True):
    return resolve_option_args(args, require_futures=args.model == 'black76', validate=validate)

NOW=datetime(2026,9,11,13,58,tzinfo=ZoneInfo('Asia/Shanghai'))
EXPIRY=date(2026,10,16)
NS=int(pd.Timestamp(NOW).value)


def candidate(name,kind='C',delta=.275,expiry=EXPIRY,**fields):
    t=years_to_expiry(NOW,expiry)
    d1=NormalDist().inv_cdf(delta if kind=='C' else 1-delta)
    strike=100*exp((.02+.25**2/2)*t-d1*.25*sqrt(t))
    option=Option(name,kind,strike,date(2026,1,1),expiry,.2,100,expiry.strftime('%Y%m'))
    close=price(100,strike,t,.02,0,.25,kind,'bs')
    quote=dict(close=close,open_interest=100,volume=20,bar_ns=NS)
    quote.update(fields)
    return option,quote


def evaluate(entries,config=Config(),forwards=None):
    options={o.symbol:o for o,q in entries}
    quotes={o.symbol:q for o,q in entries}
    return select(options,quotes,100,forwards or {},NOW,config)


class MathTests(unittest.TestCase):
    def test_iv_and_greeks_for_both_models_and_sides(self):
        for model in ('bs','black76'):
            for kind in ('C','P'):
                with self.subTest(model=model,kind=kind):
                    x,k,t,r,q,vol=6500,6600,.1,.02,.01,.25
                    premium=price(x,k,t,r,q,vol,kind,model)
                    g=implied_greeks(x,k,t,r,q,premium,kind,model)
                    self.assertAlmostEqual(g.iv,vol,places=9)
                    h=.01
                    delta=(price(x+h,k,t,r,q,vol,kind,model)-price(x-h,k,t,r,q,vol,kind,model))/(2*h)
                    self.assertAlmostEqual(delta,g.delta,places=7)
                    gamma=(price(x+h,k,t,r,q,vol,kind,model)-2*premium+price(x-h,k,t,r,q,vol,kind,model))/h**2
                    self.assertAlmostEqual(gamma,g.gamma,places=6)
                    vega=(price(x,k,t,r,q,vol+1e-5,kind,model)-price(x,k,t,r,q,vol-1e-5,kind,model))/2e-5
                    self.assertAlmostEqual(vega,g.vega,places=4)
                    self.assertIsNone(implied_greeks(x,k,t,r,q,1e9,kind,model))

    def test_put_call_parity(self):
        for model in ('bs','black76'):
            c=price(100,105,.1,.02,.01,.25,'C',model)
            p=price(100,105,.1,.02,.01,.25,'P',model)
            expected=(100 if model=='black76' else 100*exp((.02-.01)*.1))*exp(-.02*.1)-105*exp(-.02*.1)
            self.assertAlmostEqual(c-p,expected,places=10)


class SelectionTests(unittest.TestCase):
    def test_missing_nearest_month_bars_does_not_select_available_other_month(self):
        october,quote=candidate('october')
        november,nov_quote=candidate('november',expiry=date(2026,11,20))
        chosen,audit,month=select({'october':october,'november':november},
            {'november':nov_quote},100,{},NOW,replace(Config(),dte_max=80))
        self.assertEqual(chosen,[])
        self.assertEqual(month,'202610')
        self.assertIn('MISSING_BAR',[row['reason'] for row in audit])

    def test_negative_liquidity_is_invalid_even_with_zero_thresholds(self):
        config=replace(Config(),min_open_interest=0,min_volume=0)
        for field in ('open_interest','volume'):
            with self.subTest(field=field):
                chosen,audit,_=evaluate([candidate('bad',**{field:-1})],config)
                self.assertEqual(chosen,[])
                self.assertIn('INVALID_BAR_FIELDS',[row['reason'] for row in audit])

    def test_nonfinite_timing_configuration_is_rejected(self):
        for field in ('dte_min','dte_max','min_remaining_days','max_quote_age_seconds'):
            for value in (float('nan'),float('inf')):
                with self.subTest(field=field,value=value):
                    with self.assertRaisesRegex(ValueError,'有限数值'):
                        replace(Config(),**{field:value})

    def test_both_sides_otm_same_month_and_nearest_delta(self):
        chosen,_,month=evaluate([candidate('far',delta=.29),candidate('call'),candidate('put','P')])
        self.assertEqual([r['symbol'] for r in chosen],['call','put'])
        self.assertEqual(month,'202610')
        self.assertGreater(chosen[0]['delta'],0)
        self.assertLess(chosen[1]['delta'],0)
        self.assertTrue(all(not r['fallback'] for r in chosen))

    def test_fallback_and_strict_interval(self):
        entries=[candidate('fallback',delta=.24),candidate('outside',delta=.20)]
        chosen,_,_=evaluate(entries)
        self.assertEqual(chosen[0]['symbol'],'fallback')
        self.assertTrue(chosen[0]['fallback'])
        self.assertEqual(evaluate(entries,replace(Config(),allow_fallback=False))[0],[])

    def test_expiry_is_chosen_before_quality_no_cross_month_replacement(self):
        bad=candidate('bad',open_interest=0)
        chosen,audit,month=evaluate([bad,candidate('nov',expiry=date(2026,11,20))],replace(Config(),dte_max=80))
        self.assertEqual(chosen,[])
        self.assertEqual(month,'202610')
        self.assertIn('LOW_LIQUIDITY',[r['reason'] for r in audit])

    def test_quality_rejections(self):
        for changes,reason in ((dict(bar_ns=NS-60_000_000_000),'MISMATCHED_BAR_TIME'),
            (dict(bar_ns=NS+60_000_000_000),'MISMATCHED_BAR_TIME'),
            (dict(bar_ns=float('nan')),'MISSING_BAR_TIME'),
            (dict(close=0),'INVALID_CLOSE'),(dict(close=-1),'INVALID_CLOSE'),
            (dict(open_interest=0),'LOW_LIQUIDITY'),
            (dict(close=float('nan')),'MISSING_BAR_FIELDS'),
            (dict(close=float('inf')),'MISSING_BAR_FIELDS'),
            (dict(close=150),'INVALID_IV_OR_ARBITRAGE_BOUND')):
            with self.subTest(reason=reason):
                chosen,audit,_=evaluate([candidate('bad',**changes)])
                self.assertEqual(chosen,[])
                self.assertIn(reason,[r['reason'] for r in audit])

    def test_ties_follow_oi_volume_and_symbol(self):
        # 收盘价、行权价和模型相同则 Delta 相同，继续比较流动性条件。
        for poorer,better in ((dict(open_interest=10),dict(open_interest=20)),
            (dict(volume=5),dict(volume=10)),({},{})):
            chosen,_,_=evaluate([candidate('z',**poorer),candidate('a',**better)])
            self.assertEqual(chosen[0]['symbol'],'a')

    def test_black76_requires_same_month_forward(self):
        chosen,audit,_=evaluate([candidate('c')],replace(Config(),model='black76'),{'202611':100})
        self.assertEqual(chosen,[])
        self.assertEqual(audit[0]['reason'],'MISSING_SAME_MONTH_FORWARD')
        self.assertTrue(evaluate([candidate('c')],replace(Config(),model='black76'),{'202610':100})[0])

    def test_minimum_natural_days_and_fixed_month(self):
        entry=candidate('c',expiry=date(2026,9,18))
        self.assertTrue(evaluate([entry],replace(Config(),expiry_month='202609'))[0])
        self.assertEqual(evaluate([entry],replace(Config(),expiry_month='202609',min_remaining_days=8))[0],[])
        self.assertEqual(evaluate([entry])[2],None)

    def test_missing_index_and_bar(self):
        option,quote=candidate('c')
        chosen,audit,_=select({'c':option},{'c':quote},None,{},NOW,Config())
        self.assertEqual(audit[0]['reason'],'MISSING_INDEX')
        chosen,audit,_=select({'c':option},{},100,{},NOW,Config())
        self.assertIn('MISSING_BAR',[r['reason'] for r in audit])

    def test_close_is_used_even_when_order_book_disagrees(self):
        chosen,_,_=evaluate([candidate('c',bid=150,ask=151,bid_size=0,ask_size=0)])
        self.assertAlmostEqual(chosen[0]['iv'],.25,places=9)
        self.assertAlmostEqual(chosen[0]['delta'],.275,places=9)
        self.assertEqual(chosen[0]['price_source'],'close')
        for field in ('bid','ask','mid','relative_spread','depth','quote_ns'):
            self.assertNotIn(field,chosen[0])

    def test_zero_volume_is_flagged_or_filtered_by_minimum_volume(self):
        entries=[candidate('c',volume=0)]
        chosen,_,_=evaluate(entries)
        self.assertTrue(chosen[0]['zero_volume_bar'])
        chosen,audit,_=evaluate(entries,replace(Config(),min_volume=1))
        self.assertEqual(chosen,[])
        self.assertIn('LOW_LIQUIDITY',[row['reason'] for row in audit])


class InputTests(unittest.TestCase):
    def test_held_month_keeps_later_bars_outside_entry_window(self):
        option,_=candidate('C')
        first,last=date(2026,10,9),date(2026,10,12)
        args=Namespace(product='MO',model='bs',index_code='000852',future_product='IM',
            start_day=first,end_day=last,opt_basic='unused',opt_dir='options',
            index_dir='indexes',bar_timestamp='end')
        config=replace(Config(),expiry_month='202610')
        with patch.object(entry,'load_options',return_value={'C':option}), \
            patch.object(entry,'inventory',side_effect=[
                {(first,'C'):Path('first.feather'),(last,'C'):Path('last.feather')},
                {(first,'000852'):Path('first_index.feather'),(last,'000852'):Path('last_index.feather')}]), \
            patch.object(entry,'prepare_option_chain',return_value=types.SimpleNamespace(sources=())) as prepare:
            entry.prepare_inputs(args,config)
        identities={(key.trading_day,key.symbol) for key in prepare.call_args.args[1].required}
        self.assertIn((last,'C'),identities)
        self.assertLess((option.expiry-last).days,config.min_remaining_days)

    def test_invalid_option_dates_and_inconsistent_month_terms(self):
        row=dict(symbol='MO2610-C-7000',exchangeCD='CCFX',contractType='CO',
            strikePrice=7000,contMultNum=100,varTicker='000852',listDate='2026-08-01',
            lastTradeDate='2026-10-16',expDate='2026-10-16',tickNum=.2)
        for field,value,reason in (('listDate',None,'日期无效'),
            ('lastTradeDate','invalid','日期无效'),('expDate',None,'日期无效'),
            ('lastTradeDate','2026-09-18','月份与到期日期'),
            ('expDate','2026-11-20','月份与到期日期')):
            with self.subTest(field=field,value=value):
                with patch('pandas.read_feather',return_value=pd.DataFrame([{**row,field:value}])):
                    with self.assertRaisesRegex(ValueError,reason): load_options('unused','MO','000852')
        other={**row,'symbol':'MO2610-P-7000','contractType':'PO','lastTradeDate':'2026-10-15'}
        for rows in ([row,other],[other,row]):
            with patch('pandas.read_feather',return_value=pd.DataFrame(rows)):
                with self.assertRaisesRegex(ValueError,'同月合约最后交易日期'):
                    load_options('unused','MO','000852')

    def test_minute_alignment_and_cross_date_shift(self):
        for stamp,mode,reason in (('2026-09-11 13:58:30','end','整分钟'),
            ('2026-09-11 13:57:00.001','start','整分钟'),
            ('2026-09-11 23:59','start','跨日期')):
            with self.subTest(stamp=stamp,mode=mode):
                frame=pd.DataFrame([dict(datetime=stamp,close=100)])
                with patch('pandas.read_feather',return_value=frame):
                    with self.assertRaisesRegex(ValueError,reason):
                        read_frame('unused','000852',NOW.date(),'index',mode)

    def test_user_opt_basic_without_product_code(self):
        frame=pd.DataFrame([dict(symbol='MO2610-C-7000',exchangeCD='CCFX',contractType='CO',
            strikePrice=7000,contMultNum=100,varTicker='000852',listDate='2026-08-01',
            lastTradeDate='2026-10-16',expDate='2026-10-16')])
        with patch('pandas.read_feather',return_value=frame):
            with self.assertRaisesRegex(ValueError,'tickNum'): load_options('unused','MO','000852')
        frame['tickNum'] = .2
        with patch('pandas.read_feather',return_value=frame):
            self.assertEqual(load_options('unused','MO','000852')['MO2610-C-7000'].tick,.2)
        frame['tickNum'] = 0
        with patch('pandas.read_feather',return_value=frame):
            with self.assertRaisesRegex(ValueError,'tick'): load_options('unused','MO','000852')

    def test_ohlcv_without_order_book_is_accepted(self):
        frame=pd.DataFrame([dict(symbol='c',datetime='2026-09-11 13:58',close=2,volume=0,open_interest=3234)])
        with patch('pandas.read_feather',return_value=frame):
            result=read_frame('unused','C',NOW.date(),'option','end')
        self.assertEqual(result.iloc[0]['close'],2)
        self.assertEqual(result.iloc[0]['bar_ns'],NS)
        self.assertFalse({'bid','ask','bid_size','ask_size','quote_ns'} & set(result.columns))

    def test_start_time_shift_uses_close_and_ignores_quote_time(self):
        frame=pd.DataFrame([dict(symbol='MO2610-C-7000',datetime='2026-09-11 13:57',quote_datetime='2026-09-11 13:56',
            close=7,bidPrice1=1,askPrice1=2,bidVolume1=10,askVolume1=12,open_interest=100,volume=5)])
        with patch('pandas.read_feather',return_value=frame):
            row=read_frame('unused','MO2610-C-7000',NOW.date(),'option','start').iloc[0]
            self.assertEqual(row.datetime,pd.Timestamp(NOW))
            self.assertEqual(row.bar_ns,NS)
            self.assertEqual(row.close,7)

    def test_microsecond_feather_time_is_normalized_to_nanoseconds(self):
        frame=pd.DataFrame([dict(datetime=pd.Timestamp(NOW),close=2,volume=5,open_interest=100)])
        frame['datetime']=frame.datetime.dt.as_unit('us')
        with patch('pandas.read_feather',return_value=frame):
            row=read_frame('unused','C',NOW.date(),'option','end').iloc[0]
        self.assertEqual(row.bar_ns,NS)

    def test_close_is_required_even_if_order_book_exists(self):
        frame=pd.DataFrame([dict(datetime=pd.Timestamp(NOW),bidPrice1=1,askPrice1=2,volume=5,open_interest=100)])
        with patch('pandas.read_feather',return_value=frame):
            with self.assertRaisesRegex(ValueError,'close'):
                read_frame('unused','C',NOW.date(),'option','end')

    def test_bad_option_values_are_preserved_for_selection_audit(self):
        frame=pd.DataFrame([dict(datetime=pd.Timestamp(NOW),close='bad',volume=5,open_interest=100)])
        with patch('pandas.read_feather',return_value=frame):
            row=read_frame('unused','C',NOW.date(),'option','end').iloc[0]
        option,_=candidate('C')
        chosen,audit,_=select({'C':option},{'C':row.to_dict()},100,{},NOW,Config())
        self.assertEqual(chosen,[])
        self.assertIn('MISSING_BAR_FIELDS',[r['reason'] for r in audit])

    def test_existing_environment_and_explicit_overrides(self):
        args=Namespace(data_root=None,opt_dir=None,index_dir=None,fut_dir=None,opt_basic=None,fut_basic=None,model='bs')
        with patch.dict('os.environ',{'CTP_DATA_DIR':'/data','FUT_ROLE_DATA_DIR':'/roles','KLINE_DATA_DIR':'/bars'},clear=True):
            result=resolve_paths(args,validate=False)
        self.assertEqual(result.opt_dir,Path('/bars/opt'))
        self.assertEqual(result.opt_basic,Path('/roles/opt_basic.feather'))
        args.opt_dir=Path('/chosen')
        with patch.dict('os.environ',{},clear=True):
            self.assertEqual(resolve_paths(args,validate=False).opt_dir,Path('/chosen'))


class FrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        package=types.ModuleType('bomber.framework.trader'); package.__path__=[str(root/'bomber/framework/trader')]
        execution=types.ModuleType('bomber.framework.trader.execution'); execution.__path__=[str(root/'bomber/framework/trader/execution')]
        market=types.ModuleType('bomber.framework.market.basic.base')
        from enum import Enum
        market.DataType=Enum('DataType','BAR CUSTOM_BAR TRADE_TICK QUOTE_TICK')
        market.InstrumentId=str
        for name in ('Bar','CustomBar','TradeTick','QuoteTick'): setattr(market,name,type(name,(),{}))
        with patch.dict(sys.modules,{'bomber.framework.trader':package,'bomber.framework.trader.execution':execution,'bomber.framework.market.basic.base':market}):
            cls.Module = import_module('demos.09_option_delta.strategy')
            # 在模块夹具恢复前保存真实模板使用的事件类型。
            cls.Event=sys.modules['bomber.framework.trader.template'].CustomBar
        cls.Strategy=cls.Module.DeltaSelectionRecorder

    def strategy(self):
        entries=[candidate('c'),candidate('p','P')]
        strategy=self.Strategy('test',Config(),{o.symbol:o for o,q in entries},{})
        strategy._bind(types.SimpleNamespace(submit=lambda intent:self.fail('selection must not submit orders')))
        strategy._start()
        for key,factors in [('000852',{'close':100})]+[(o.symbol,q) for o,q in entries]:
            event=self.Event(); event.ts_event=NS; event.factors=factors
            strategy._handle_event(key,event)
        return strategy

    def test_complete_frame_at_next_timestamp_and_no_orders(self):
        strategy=self.strategy()
        self.assertEqual(strategy.records,[])
        event=self.Event(); event.ts_event=NS+60_000_000_000; event.factors={'close':101}
        strategy._handle_event('000852',event)
        self.assertEqual(strategy.records[0]['status'],'SELECTED')
        self.assertEqual(len(strategy.selections),2)
        self.assertEqual(strategy.selections[0]['index'],100)
        strategy._stop()
        self.assertEqual(len(strategy.records),1)

    def test_stop_flushes_last_complete_selection_frame(self):
        strategy=self.strategy(); strategy._stop()
        self.assertEqual(len(strategy.selections),2)

    def test_duplicate_and_out_of_order_snapshots_rejected(self):
        strategy=self.strategy()
        event=self.Event(); event.ts_event=NS; event.factors={'close':100}
        with self.assertRaisesRegex(ValueError,'重复'): strategy._handle_event('000852',event)
        event.ts_event=NS-1
        with self.assertRaisesRegex(ValueError,'倒退'): strategy._handle_event('000852',event)

    def test_large_gap_is_reported_without_using_next_frame(self):
        strategy=self.strategy()
        event=self.Event(); event.ts_event=NS+120_000_000_000; event.factors={'close':100}
        strategy._handle_event('000852',event)
        self.assertEqual(strategy.records[0]['status'],'LATE_FRAME')
        self.assertEqual(strategy.selections,[])

    def test_subminute_snapshot_cannot_trigger_selection_slot(self):
        strategy=self.strategy()
        event=self.Event(); event.ts_event=NS+30_000_000_000; event.factors={'close':100}
        with self.assertRaisesRegex(ValueError,'整分钟'):
            strategy._handle_event('000852',event)
        self.assertEqual(strategy.records,[])

    def test_replay_entry_preserves_month_with_no_files(self):
        october,_=candidate('OCTOBER')
        november,quote=candidate('NOVEMBER',expiry=date(2026,11,20))
        options={'OCTOBER':october,'NOVEMBER':november}
        args=Namespace(product='MO',model='bs',index_code='000852',future_product='IM',
            start_day=NOW.date(),end_day=NOW.date(),opt_basic='unused',opt_dir='options',
            index_dir='indexes',bar_timestamp='end')
        config=replace(Config(),dte_max=80)
        with patch.object(entry,'load_options',return_value=options), \
            patch.object(entry,'inventory',side_effect=[
                {(NOW.date(),'NOVEMBER'):Path('november.feather')},
                {(NOW.date(),'000852'):Path('index.feather')}]), \
            patch.object(entry,'prepare_option_chain',return_value=types.SimpleNamespace(sources=())) as prepare:
            loaded,_,_,days,_=entry.prepare_inputs(args,config)
        self.assertEqual(set(loaded),{'OCTOBER','NOVEMBER'})
        self.assertEqual(days,(NOW.date(),))
        plan=prepare.call_args.args[1]
        self.assertEqual({key.symbol for key in plan.required},{'000852','NOVEMBER'})
        self.assertEqual(prepare.call_args.kwargs['specs']['option'].value_policy,'execution_strict')
        chosen,audit,month=select(loaded,{'NOVEMBER':quote},100,{},NOW,config)
        self.assertEqual(chosen,[])
        self.assertEqual(month,'202610')
        self.assertIn('MISSING_BAR',[row['reason'] for row in audit])


class ParserTests(unittest.TestCase):
    def test_execution_parser_emits_one_real_bar_then_custom_factors(self):
        FrameTests.setUpClass()
        module=FrameTests.Module
        base=types.ModuleType('bomber.framework.market.basic.base')
        base.DataType=types.SimpleNamespace(BAR='bar',CUSTOM_BAR='custom')
        base.make_bar=lambda *args,**kwargs:self.fail('交易分支不能构造替代价格')
        base.make_custom_bar=lambda bar,factors:types.SimpleNamespace(bar=bar,factors=factors)
        parsed=types.ModuleType('bomber.framework.market.replay.parsers.base')
        parsed.ParsedEvent=lambda dtype,item,event,bar_type:types.SimpleNamespace(data_type=dtype,event=event)
        real_bar=types.SimpleNamespace(close=7,ts_event=NS,ts_init=NS)
        native=types.SimpleNamespace(data_type='bar',payload=real_bar)
        parser=module.SnapshotParser('C.CFFEX')
        parser.execution_parser=types.SimpleNamespace(parse=lambda row,context:iter((native,)))
        row=dict(datetime=pd.Timestamp(NOW),available_ns=NS,close=7,volume=1,open_interest=100,bar_ns=NS)
        with patch.dict(sys.modules,{'bomber.framework.market.basic.base':base,'bomber.framework.market.replay.parsers.base':parsed}):
            events=list(parser.parse(row,types.SimpleNamespace()))
        self.assertEqual(len(events),2)
        self.assertIs(events[0],native)
        self.assertEqual(events[1].data_type,'custom')
        self.assertIs(events[1].event.bar,real_bar)
        self.assertEqual(events[1].event.factors['close'],7)

    def test_parser_carries_close_without_order_book_and_preserves_invalid_price(self):
        base=types.ModuleType('bomber.framework.market.basic.base')
        base.DataType=types.SimpleNamespace(CUSTOM_BAR='custom')
        base.make_bar=lambda item,o,h,l,c,volume,event_ns,init_ns,**kwargs:types.SimpleNamespace(
            close=c,ts_event=event_ns,ts_init=init_ns)
        base.make_custom_bar=lambda bar,factors:types.SimpleNamespace(bar=bar,factors=factors)
        parsed=types.ModuleType('bomber.framework.market.replay.parsers.base')
        parsed.ParsedEvent=lambda dtype,item,event,bar_type:types.SimpleNamespace(event=event)
        # 复用真实模板与事件类型夹具，不依赖本机原生运行时。
        FrameTests.setUpClass()
        module=FrameTests.Module
        parser=module.SnapshotParser('C.CFFEX')
        context=types.SimpleNamespace(require_meta=lambda item:None)
        for close in (7,float('nan')):
            with self.subTest(close=close):
                row=dict(datetime=pd.Timestamp(NOW),close=close,open_interest=100,volume=0,bar_ns=NS,
                    bid=150,ask=151,bidPrice1=150,askPrice1=151,quote_ns=NS)
                with patch.dict(sys.modules,{'bomber.framework.market.basic.base':base,'bomber.framework.market.replay.parsers.base':parsed}):
                    event=next(parser.parse(row,context)).event
                self.assertEqual(set(event.factors),{'close','open_interest','volume','bar_ns'})
                self.assertEqual(event.bar.ts_event,NS)
                if pd.isna(close):
                    self.assertTrue(pd.isna(event.factors['close']))
                    self.assertEqual(event.bar.close,1)
                else:
                    self.assertEqual(event.factors['close'],7)
                    self.assertEqual(event.bar.close,7)



class TradingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FrameTests.setUpClass()
        cls.Strategy=FrameTests.Module.OptionDeltaStrategy

    def setUp(self):
        from decimal import Decimal
        self.Decimal=Decimal
        self.options={name:candidate(name,kind)[0] for name,kind in
                      (('c','C'),('p','P'),('oldc','C'),('oldp','P'))}
        self.ids={key:key+'.CFFEX' for key in self.options}
        self.actual={}
        self.working={}
        self.intents=[]
        self.strategy=self.Strategy('trade',Config(),self.options,{},
            instruments=self.ids,final_day=date(2026,9,15))
        self.strategy._bind(types.SimpleNamespace(submit=self.intents.append,
            account_position=lambda key:self.actual.get(key,Decimal(0)),
            working_quantity=lambda key:self.working.get(key,Decimal(0))))
        self.strategy._start()
        self.strategy.latest={key:dict(event_ns=NS,close=5) for key in self.options}
        self.chosen=[dict(symbol='c',kind='C'),dict(symbol='p',kind='P')]

    def choose(self, status='SELECTED', chosen=None):
        self.strategy._after_selection(dict(status=status,signal_ns=NS),
            self.chosen if chosen is None else chosen,NS+60_000_000_000)

    def drive(self):
        self.strategy._drive_targets(NS,NS+60_000_000_000)

    def test_complete_pair_buys_each_leg_and_does_not_accumulate(self):
        self.choose(); self.drive()
        targets=self.intents[-1].targets
        self.assertEqual(targets[self.ids['c']],1)
        self.assertEqual(targets[self.ids['p']],1)
        self.assertEqual(targets[self.ids['oldc']],0)
        self.actual={self.ids['c']:self.Decimal(1),self.ids['p']:self.Decimal(1)}
        self.drive(); self.choose(); self.drive()
        self.assertEqual(len(self.intents),1)

    def test_partial_selection_does_not_buy_single_leg(self):
        self.choose('PARTIAL',self.chosen[:1]); self.drive()
        self.assertEqual(self.intents,[])
        self.assertIsNone(self.strategy.pending_targets)

    def test_roll_closes_old_pair_before_opening_new_pair(self):
        for key in ('oldc','oldp'):
            self.actual[self.ids[key]]=self.Decimal(1)
            self.strategy.goals[key]=self.Decimal(1)
        self.choose(); self.drive()
        self.assertTrue(all(value==0 for value in self.intents[-1].targets.values()))
        self.working[self.ids['oldc']]=self.Decimal(-1)
        self.drive()
        self.assertEqual(len(self.intents),1)
        self.actual.clear(); self.working.clear(); self.drive()
        self.assertEqual(self.intents[-1].targets[self.ids['c']],1)
        self.assertEqual(self.intents[-1].targets[self.ids['p']],1)

    def test_stale_old_leg_waits_without_clearing_actual_position(self):
        self.actual[self.ids['oldc']]=self.Decimal(1)
        self.strategy.goals['oldc']=self.Decimal(1)
        self.strategy.latest['oldc']['event_ns']=NS-600_000_000_000
        self.choose(); self.drive()
        self.assertEqual(self.intents,[])
        self.assertEqual(self.strategy.signals[-1]['action'],'WAIT_FRESH_PRICES')
        self.assertEqual(self.actual[self.ids['oldc']],1)
        self.assertEqual(self.strategy.goals['oldc'],1)
        self.strategy.latest['oldc']['event_ns']=NS
        self.drive()
        self.assertTrue(all(value==0 for value in self.intents[-1].targets.values()))

    def test_final_exit_submits_zero_and_stop_never_opens(self):
        stamp=int(pd.Timestamp('2026-09-15 14:50',tz='Asia/Shanghai').value)
        for key in ('c','p'):
            self.actual[self.ids[key]]=self.Decimal(1)
            self.strategy.goals[key]=self.Decimal(1)
            self.strategy.latest[key]['event_ns']=stamp
        self.strategy._evaluate(stamp,stamp+60_000_000_000)
        self.assertTrue(self.strategy.flatten_requested)
        self.assertTrue(all(value==0 for value in self.intents[-1].targets.values()))
        self.strategy.pending_ns=int(pd.Timestamp('2026-09-15 13:58',tz='Asia/Shanghai').value)
        count=len(self.intents)
        self.strategy._stop()
        self.assertEqual(len(self.intents),count)
        self.assertEqual(self.strategy.records[-1]['status'],'NO_FOLLOWING_FRAME')

    def test_expiry_exit_closes_both_legs(self):
        for key in ('c','p'):
            self.strategy.options[key]=replace(self.options[key],expiry=date(2026,9,13))
            self.actual[self.ids[key]]=self.Decimal(1)
            self.strategy.goals[key]=self.Decimal(1)
        self.strategy._evaluate(NS+60_000_000_000,NS+120_000_000_000)
        self.assertFalse(self.strategy.flatten_requested)
        self.assertEqual(self.strategy.signals[-1]['action'],'PRE_EXPIRY_FLAT')
        self.assertTrue(all(value==0 for value in self.intents[-1].targets.values()))

    def test_rejected_order_fails(self):
        with self.assertRaisesRegex(RuntimeError,'REJECTED'):
            self.strategy.on_order_update(types.SimpleNamespace(status=types.SimpleNamespace(value='REJECTED'),
                identity=types.SimpleNamespace(instrument_id=self.ids['c']),reason='fixture'))

    def test_invalid_trade_parameters(self):
        for params in ({'quantity':0},{'quantity':1.5},{'close_remaining_days':7},
                       {'max_market_age_seconds':float('nan')}):
            with self.subTest(params=params):
                with self.assertRaises(ValueError): replace(Config(),**params)

    def test_single_side_trading_rejected(self):
        with self.assertRaisesRegex(ValueError,'C 和 P'):
            self.Strategy('bad',replace(Config(),sides=('C',)),self.options,{},
                instruments=self.ids,final_day=date(2026,9,15))



class ContractFactoryTests(unittest.TestCase):
    def test_call_and_put_identity_and_default_call_compatibility(self):
        import importlib.util
        from decimal import Decimal
        path=Path(__file__).resolve().parents[1]/'bomber/framework/trader/instrument_factory.py'
        spec=importlib.util.spec_from_file_location('delta_contract_factory_fixture',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        constructors=types.SimpleNamespace(from_str=lambda value:value,from_int=lambda value:value)
        base=types.ModuleType('bomber.model')
        base.Price=base.Quantity=constructors
        base.Symbol=lambda value:value
        enums=types.ModuleType('bomber.model.enums')
        enums.AssetClass=types.SimpleNamespace(INDEX='INDEX')
        enums.OptionKind=types.SimpleNamespace(CALL='CALL',PUT='PUT')
        currencies=types.ModuleType('bomber.model.currencies'); currencies.CNY='CNY'
        identifiers=types.ModuleType('bomber.model.identifiers'); identifiers.InstrumentId=constructors
        instruments=types.ModuleType('bomber.model.instruments'); instruments.OptionContract=lambda **values:values
        info=types.SimpleNamespace(symbol='MO2610-C-7000',strike=7000,tick=Decimal('.2'),
            multiplier=Decimal(100),list_day=date(2026,8,1),last_day=EXPIRY)
        with patch.dict(sys.modules,{'bomber.model':base,'bomber.model.enums':enums,
            'bomber.model.currencies':currencies,'bomber.model.identifiers':identifiers,
            'bomber.model.instruments':instruments}):
            default=module.make_option(info,Decimal(1),Decimal(1),'000852')
            self.assertEqual(default['option_kind'],'CALL')
            self.assertEqual(default['info']['profile'],'OPTION_VEGA_BASIC')
            info.symbol='MO2610-P-7000'
            put=module.make_option(info,Decimal(1),Decimal(1),'000852',kind='P',profile_id='OPTION_DELTA_BASIC')
            self.assertEqual(put['option_kind'],'PUT')
            self.assertEqual(put['instrument_id'],'MO2610-P-7000.CFFEX')
            self.assertEqual(put['info']['profile'],'OPTION_DELTA_BASIC')
        with self.assertRaises(ValueError): module.make_option(info,1,1,'000852',kind='X')


if __name__=='__main__': unittest.main()
