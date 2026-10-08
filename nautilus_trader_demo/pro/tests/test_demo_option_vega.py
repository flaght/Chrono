"""数值、Feather资料和真实StrategyTemplate目标接口夹具；不假装原生撮合。"""
from datetime import date
from argparse import Namespace
from decimal import Decimal
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd
from importlib import import_module
from functools import partial

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "demos" / "08_option_vega"))
from vega_signal import (Contract, OptionVegaConfig, remaining_trading_days,
                         allocate_short, call_price, implied_greeks, filter_hedgeable_options)
from bomber.framework.dataprep.scenarios.option_chain import read_execution_frame as read_frame
from bomber.framework.dataprep.calendar import load_calendar, infer_market_calendar
from bomber.framework.dataprep.catalog import inventory
from bomber.framework.dataprep.paths import resolve_option_args
load_contracts = import_module("demos.08_option_vega.run_backtest").load_contracts
resolve_paths = partial(resolve_option_args, require_futures=True)


class Price:
    def __init__(self, value):
        self.value = Decimal(str(value))
    def as_decimal(self):
        return self.value


class Bar:
    def __init__(self, timestamp, price):
        self.ts_event = timestamp
        self.close = Price(price)


class Context:
    def __init__(self):
        self.intents = []
        self.actual = {}
        self.working = {}
    def submit(self, intent):
        self.intents.append(intent)
    def position(self, key):
        return self.account_position(key)
    def account_position(self, key):
        return self.actual.get(key, Decimal(0))
    def working_quantity(self, key):
        return self.working.get(key, Decimal(0))


def ns(value):
    return int(pd.Timestamp(value, tz="Asia/Shanghai").value)


def strategy_class():
    # 不安装假交易引擎；只替换行情扩展类型，读取项目真实template/contracts/events。
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType("bomber.framework.trader")
    package.__path__ = [str(root / "bomber" / "framework" / "trader")]
    execution = types.ModuleType("bomber.framework.trader.execution")
    execution.__path__ = [str(root / "bomber" / "framework" / "trader" / "execution")]
    market = types.ModuleType("bomber.framework.market.basic.base")
    from enum import Enum
    market.DataType = Enum("DataType", "BAR CUSTOM_BAR TRADE_TICK QUOTE_TICK")
    market.InstrumentId = str
    market.Bar = Bar
    for name in ("CustomBar", "TradeTick", "QuoteTick"):
        setattr(market, name, type(name, (), {}))
    with patch.dict(sys.modules, {"bomber.framework.trader": package, "bomber.framework.trader.execution": execution, "bomber.framework.market.basic.base": market}):
        OptionVegaStrategy = import_module("demos.08_option_vega.strategy").OptionVegaStrategy
        return OptionVegaStrategy


class SignalTests(unittest.TestCase):
    def test_direct_script_directory_does_not_shadow_stdlib_signal(self):
        directory=Path(__file__).resolve().parents[1]/"demos"/"08_option_vega"
        result=subprocess.run([sys.executable,"-c",
            "import signal; assert hasattr(signal, 'NSIG'), signal.__file__; "
            "assert hasattr(signal, 'SIGINT'), signal.__file__"],
            cwd=directory,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_iv_roundtrip_and_discounted_derivatives(self):
        f, k, t, r, iv = 6500, 6600, 0.1, 0.02, 0.25
        price = call_price(f,k,t,r,iv)
        greeks = implied_greeks(f,k,t,r,price)
        self.assertAlmostEqual(greeks.iv, iv, places=9)
        eps = 0.01
        delta = (call_price(f+eps,k,t,r,iv)-call_price(f-eps,k,t,r,iv))/(2*eps)
        self.assertAlmostEqual(delta, greeks.delta, places=7)
        vega = (call_price(f,k,t,r,iv+1e-5)-call_price(f,k,t,r,iv-1e-5))/2e-5
        self.assertAlmostEqual(vega, greeks.vega, places=4)
        self.assertIsNone(implied_greeks(f,k,t,r,f*2))
        self.assertIsNone(implied_greeks(f,k,t,r,float("nan")))

    def test_remaining_days_exact_and_far_expiry_lower_bound(self):
        days = tuple(stamp.date() for stamp in pd.bdate_range("2026-09-01", "2026-09-10"))
        self.assertEqual(remaining_trading_days(days,date(2026,9,1),date(2026,9,4),5),3)
        # 已知日期只到9月10日，但足够判断更远到期仍有至少5个交易日。
        self.assertEqual(remaining_trading_days(days,date(2026,9,1),date(2026,12,18),5),7)
        # 日期尾部不能将未知未来误判为0个交易日而提前清仓。
        with self.assertRaisesRegex(ValueError,"无法判断"):
            remaining_trading_days(days,date(2026,9,9),date(2026,12,18),3)

    def test_duplicate_delta_targets_budget_and_lot_caps(self):
        g = implied_greeks(6500,6600,0.1,0.02,call_price(6500,6600,0.1,0.02,0.25))
        quantity = allocate_short([("x",g,100)],(0.2,0.25,0.3,0.35),500000,100,200)
        self.assertLessEqual(abs(quantity["x"])*g.vega*100,500000)
        self.assertLessEqual(abs(quantity["x"]),100)
        small = allocate_short([("x",g,100)],(0.25,),1,100,200)
        self.assertEqual(small,{})


class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Strategy = strategy_class()

    def setUp(self):
        self.info = Contract("MO2609-C-6600","IM2609",6600,Decimal(100),date(2026,6,1),date(2026,9,18),date(2026,9,18),Decimal("0.2"))
        self.days = tuple(stamp.date() for stamp in pd.bdate_range("2026-08-13","2026-09-18"))
        self.ids = {self.info.symbol:self.info.symbol+".CFFEX", "IM2609":"IM2609.CFFEX"}
        self.ctx = Context()
        self.strategy = self.Strategy("test",OptionVegaConfig(min_time_value=0),{self.info.symbol:self.info},
            {"IM2609":{"contMultNum":Decimal(200)}},self.ids,"000852",self.days,date(2026,8,14),
            {date(2026,8,13):(self.info.symbol,),date(2026,8,14):(self.info.symbol,)})
        self.strategy._bind(self.ctx)
        self.strategy._start()

    def frame(self, clock, option_price=None):
        from vega_signal import tau
        now = pd.Timestamp(clock,tz="Asia/Shanghai").to_pydatetime()
        premium = option_price if option_price is not None else call_price(6500,6600,tau(now,self.info.last_day),0.02,0.25)
        for key, price in (("000852",6400),("IM2609",6500),(self.info.symbol,premium)):
            self.strategy._handle_event(key,Bar(ns(clock),price))

    def test_frame_barrier_and_actual_fill_hedge(self):
        self.frame("2026-08-13 13:58")
        self.assertEqual(self.ctx.intents,[])
        self.frame("2026-08-13 13:59")
        entry = self.ctx.intents[-1]
        self.assertLess(entry.targets[self.ids[self.info.symbol]],0)
        self.assertEqual(entry.targets[self.ids["IM2609"]],0)
        # 只成交2手，不把全部目标数量误当实际持仓。
        self.ctx.actual[self.ids[self.info.symbol]]=Decimal(-2)
        self.frame("2026-08-13 14:00")
        hedge = self.ctx.intents[-1].targets[self.ids["IM2609"]]
        self.assertIn(hedge,(Decimal(0),Decimal(1)))
        self.assertEqual(self.strategy.entry_decisions,1)

    def test_daily_budget_replaces_instead_of_accumulates(self):
        self.frame("2026-08-13 13:58")
        self.frame("2026-08-13 13:59")
        first = abs(self.ctx.intents[-1].targets[self.ids[self.info.symbol]])
        self.ctx.actual[self.ids[self.info.symbol]]=-first
        self.frame("2026-08-14 13:58")
        self.frame("2026-08-14 13:59")
        second = abs(self.ctx.intents[-1].targets[self.ids[self.info.symbol]])
        self.assertLessEqual(second, first+2)
        self.assertEqual(self.strategy.entry_decisions,2)

    def test_final_flat_contains_explicit_zero_targets(self):
        self.ctx.actual[self.ids[self.info.symbol]]=Decimal(-2)
        self.ctx.actual[self.ids["IM2609"]]=Decimal(1)
        self.frame("2026-08-14 14:50")
        self.frame("2026-08-14 14:51")
        self.assertTrue(self.strategy.flatten_requested)
        self.assertEqual(set(self.ctx.intents[-1].targets),set(self.ids.values()))
        self.assertTrue(all(value==0 for value in self.ctx.intents[-1].targets.values()))

    def test_missing_held_price_fails_and_pending_hedge_not_reversed(self):
        self.ctx.actual[self.ids[self.info.symbol]]=Decimal(-2)
        self.strategy.latest = {"000852":(ns("2026-08-13 10:00"),6400),"IM2609":(ns("2026-08-13 10:00"),6500)}
        with self.assertRaises(RuntimeError):
            self.strategy._frame(ns("2026-08-13 10:00"),ns("2026-08-13 10:01"))
        self.strategy.latest={}
        self.ctx.working[self.ids["IM2609"]]=Decimal(1)
        self.frame("2026-08-13 10:00")
        self.frame("2026-08-13 10:01")
        self.assertEqual(self.ctx.intents,[])

    def test_final_flat_does_not_require_unheld_option_price(self):
        self.ctx.actual[self.ids["IM2609"]]=Decimal(1)
        self.strategy.future_goals["IM2609"]=Decimal(1)
        stamp=ns("2026-08-14 14:50")
        self.strategy.latest={"IM2609":(stamp,6500)}
        self.strategy._frame(stamp,ns("2026-08-14 14:51"))
        self.assertTrue(self.strategy.flatten_requested)
        self.assertEqual(self.ctx.intents[-1].targets[self.ids["IM2609"]],Decimal(0))
        self.assertEqual(self.ctx.intents[-1].targets[self.ids[self.info.symbol]],Decimal(0))

    def test_final_flat_rejects_stale_price_for_actual_position(self):
        self.ctx.actual[self.ids[self.info.symbol]]=Decimal(-1)
        self.strategy.latest={self.info.symbol:(ns("2026-08-14 14:00"),100)}
        with self.assertRaisesRegex(RuntimeError,"缺少新鲜行情"):
            self.strategy._frame(ns("2026-08-14 14:50"),ns("2026-08-14 14:51"))
        self.assertEqual(self.ctx.intents,[])

    def test_resume_waits_for_delayed_option_first_bar(self):
        self.ctx.actual[self.ids[self.info.symbol]]=Decimal(-2)
        self.strategy.option_goals[self.info.symbol]=Decimal(-2)
        self.ctx.actual[self.ids["IM2609"]]=Decimal(1)
        self.strategy.future_goals["IM2609"]=Decimal(1)
        self.frame("2026-08-13 11:30")
        self.strategy._handle_event("IM2609",Bar(ns("2026-08-13 13:00"),6500))
        self.frame("2026-08-13 13:01")
        self.assertEqual(self.ctx.intents,[])
        self.assertEqual(self.strategy.signals[-1]["action"],"RESUME_WAIT")
        self.assertEqual(self.strategy.future_goals["IM2609"],Decimal(1))
        self.frame("2026-08-13 13:02")
        self.assertIsNone(self.strategy.resume_ns)

    def test_resume_missing_held_option_still_fails_after_timeout(self):
        self.ctx.actual[self.ids[self.info.symbol]]=Decimal(-2)
        self.strategy.option_goals[self.info.symbol]=Decimal(-2)
        self.frame("2026-08-13 11:30")
        resumed = pd.Timestamp("2026-08-13 13:00")
        # 新分钟触发上一帧：先覆盖等待期限内所有分钟，包括恰好到期的帧。
        last_wait_minute = self.strategy.config.max_market_age_seconds // 60 + 1
        for minute in range(last_wait_minute + 1):
            clock = (resumed + pd.Timedelta(minutes=minute)).strftime("%Y-%m-%d %H:%M")
            self.strategy._handle_event("IM2609",Bar(ns(clock),6500))
            self.strategy._handle_event("000852",Bar(ns(clock),6400))
        self.assertEqual(self.strategy.signals[-1]["action"],"RESUME_WAIT")
        self.assertEqual(self.strategy.option_goals[self.info.symbol],Decimal(-2))
        self.assertEqual(self.ctx.actual[self.ids[self.info.symbol]],Decimal(-2))
        # 再推进一帧，使被评价的上一分钟严格超过配置等待期限。
        after_timeout = (resumed + pd.Timedelta(minutes=last_wait_minute + 1)).strftime("%Y-%m-%d %H:%M")
        with self.assertRaisesRegex(RuntimeError,"最新报价时间=.*报价年龄秒=.*实际仓位=-2"):
            self.strategy._handle_event("IM2609",Bar(ns(after_timeout),6500))
        self.assertEqual(self.ctx.intents,[])


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def test_feather_without_timestamp_and_start_label_shift(self):
        path=self.root/"MO2609-C-6600_20260813.feather"
        pd.DataFrame(dict(symbol=["MO2609-C-6600"],trade_date=["2026-08-13"],
            datetime=pd.date_range("2026-08-13 13:58",periods=1,tz="Asia/Shanghai"),
            open=[20.0],high=[20.0],low=[20.0],close=[20.0],volume=[0])).to_feather(path)
        frame=read_frame(path,"MO2609-C-6600",date(2026,8,13),"start")
        self.assertEqual(frame.datetime.iloc[0].minute,59)
        self.assertEqual(str(frame.datetime.iloc[0].tzinfo),"Asia/Shanghai")
        self.assertNotIn("timestamp",frame)

    def test_metadata_duplicate_conflict_and_missing_tick(self):
        op=self.root/"opt_basic.feather"
        fp=self.root/"fut_basic.feather"
        row=dict(Code="MO2609-C-6600",contractType="CO",strikePrice=6600,contMultNum=100,
            varTicker="000852",exchangeCD="CCFX",listDate="2026-06-01",lastTradeDate="2026-09-18",expDate="2026-09-18")
        pd.DataFrame([row,row]).to_feather(op)
        pd.DataFrame([dict(symbol="IM2609",code="IM",exchangeCD="CCFX",contMultNum=200,minChgPriceNum=0.2,
            listDate="2025-01-01",lastTradeDate="2026-09-18")]).to_feather(fp)
        with self.assertRaises(ValueError):
            load_contracts(op,fp,"MO","IM","000852")
        row["tickNum"] = 0.2
        pd.DataFrame([row,row]).to_feather(op)
        contracts,futures=load_contracts(op,fp,"MO","IM","000852")
        self.assertEqual(len(contracts),1)
        self.assertEqual(contracts["MO2609-C-6600"].tick, Decimal("0.2"))
        self.assertEqual(futures["IM2609"]["contMultNum"],Decimal(200))
        conflicting={**row,"contMultNum":200}
        pd.DataFrame([row,conflicting]).to_feather(op)
        with self.assertRaises(ValueError):
            load_contracts(op,fp,"MO","IM","000852")

    def test_option_symbol_and_code_aliases_without_product_column(self):
        row=dict(symbol=" MO2609-C-6600  ",contractType="CO",strikePrice=6600,
            contMultNum=100,varTicker="000852",exchangeCD="CCFX",currencyCD="CNY",
            tickNum=0.2, minChgPriceNum=0.5,
            listDate="2026-06-01",lastTradeDate="2026-09-18",expDate="2026-09-18")
        future=pd.DataFrame([dict(symbol="IM2609",code="IM",exchangeCD="CCFX",
            contMultNum=200,minChgPriceNum=0.2,listDate="2025-01-01",lastTradeDate="2026-09-18")])
        for options in (pd.DataFrame([row]),pd.DataFrame([row]).rename(columns={"symbol":"Code"}),
                        pd.DataFrame([{**row,"Code":"MO2609-C-6600.CFFEX"}])):
            with patch("pandas.read_feather",side_effect=[options,future]):
                contracts,_=load_contracts("opt.feather","fut.feather","MO","IM","000852")
            self.assertEqual(contracts["MO2609-C-6600"].future,"IM2609")
            self.assertEqual(contracts["MO2609-C-6600"].tick, Decimal("0.2"))
        bad=pd.DataFrame([{**row,"Code":"MO2609-C-6700"}])
        with patch("pandas.read_feather",side_effect=[bad,future]):
            with self.assertRaisesRegex(ValueError,"Code与symbol不一致"):
                load_contracts("opt.feather","fut.feather","MO","IM","000852")

    def test_option_tick_num_validation_and_legacy_fallback(self):
        row=dict(symbol="MO2609-C-6600",contractType="CO",strikePrice=6600,
            contMultNum=100,varTicker="000852",exchangeCD="CCFX",currencyCD="CNY",
            minChgPriceNum=0.5,listDate="2026-06-01",
            lastTradeDate="2026-09-18",expDate="2026-09-18")
        future=pd.DataFrame([dict(symbol="IM2609",code="IM",exchangeCD="CCFX",
            contMultNum=200,minChgPriceNum=0.2,listDate="2025-01-01",lastTradeDate="2026-09-18")])
        # 新字段缺失或为空可以兼容旧条款；无效的非空新字段不能被旧值掩盖。
        for options in (pd.DataFrame([row]),pd.DataFrame([{**row,"tickNum":float("nan")}])):
            with patch("pandas.read_feather",side_effect=[options,future]):
                contracts,_=load_contracts("opt.feather","fut.feather","MO","IM","000852")
            self.assertEqual(contracts["MO2609-C-6600"].tick,Decimal("0.5"))
        for tick in (0,-0.2,float("inf"),"invalid"):
            with self.subTest(tick=tick):
                with patch("pandas.read_feather",return_value=pd.DataFrame([{**row,"tickNum":tick}])):
                    with self.assertRaisesRegex(ValueError,"有限正数"):
                        load_contracts("opt.feather","fut.feather","MO","IM","000852")

    def test_path_environment_keeps_role_and_market_separate(self):
        roles = self.root / "role"
        kline = self.root / "kline"
        for kind in ("fut", "opt", "index"):
            (kline / kind).mkdir(parents=True)
        args = Namespace(data_root=None, fut_dir=None, opt_dir=None, index_dir=None,
                         fut_basic=None, opt_basic=None, calendar=None)
        with patch.dict("os.environ", {"FUT_ROLE_DATA_DIR":str(roles),
                        "KLINE_DIR":str(kline), "CTP_CALENDAR_PATH":str(self.root/"calendar.csv")}, clear=True):
            resolved = resolve_paths(args, validate=False)
        self.assertEqual(resolved.opt_basic, roles / "opt_basic.feather")
        self.assertEqual(resolved.fut_dir, kline / "fut")
        self.assertEqual(resolved.opt_dir, kline / "opt")
        self.assertEqual(resolved.index_dir, kline / "index")

    def test_unlisted_hedge_is_excluded_until_listing_date(self):
        options = {}
        for month,last in (("2609",date(2026,9,18)),("2611",date(2026,11,20))):
            key="MO"+month+"-6600"
            options[key]=Contract(key,"IM"+month,6600,Decimal(100),date(2026,6,1),last,last,Decimal("0.2"))
        futures={"IM2609":{"listDate":date(2026,1,19),"lastTradeDate":date(2026,9,18)},
                 "IM2611":{"listDate":date(2026,9,21),"lastTradeDate":date(2026,11,20)}}
        eligible,skipped=filter_hedgeable_options(tuple(options),options,futures,date(2026,9,11))
        self.assertEqual(eligible,("MO2609-6600",))
        self.assertEqual(skipped,("IM2611",))
        eligible,_=filter_hedgeable_options(tuple(options),options,futures,date(2026,9,21))
        self.assertEqual(eligible,("MO2611-6600",))
        eligible,skipped=filter_hedgeable_options(tuple(options),options,{},date(2026,9,11))
        self.assertEqual(eligible,())
        self.assertEqual(skipped,("IM2609","IM2611"))

    def test_far_option_without_same_month_basic_does_not_block_near_month(self):
        keys=("MO2609-C-6600","MO2706-C-6200")
        options={key:Contract(key,future,strike,Decimal(100),date(2026,1,1),last,last,Decimal("0.2"))
                 for key,future,strike,last in ((keys[0],"IM2609",6600,date(2026,9,18)),
                                                (keys[1],"IM2706",6200,date(2027,6,18)))}
        futures={"IM2609":{"listDate":date(2026,1,19),"lastTradeDate":date(2026,9,18)}}
        eligible,skipped=filter_hedgeable_options(keys,options,futures,date(2026,9,11))
        self.assertEqual(eligible,(keys[0],))
        self.assertEqual(skipped,("IM2706",))

    def test_calendar_optional_in_environment(self):
        args = Namespace(data_root=self.root, fut_dir=None, opt_dir=None, index_dir=None,
                         fut_basic=None, opt_basic=None, calendar=None)
        with patch.dict("os.environ", {}, clear=True):
            resolved = resolve_paths(args, validate=False)
        self.assertIsNone(resolved.calendar)
        self.assertEqual(resolved.opt_dir,self.root / "kline" / "opt")

    def test_infer_calendar_uses_all_dates_and_ignores_empty_folders(self):
        roots = [self.root / kind for kind in ("fut", "opt", "index")]
        for root in roots:
            root.mkdir()
        for root, clock, code in ((roots[0],"20260901","IM2609"),
                                  (roots[1],"20260902","MO2609-C-6600"),
                                  (roots[2],"20260904","000852")):
            folder = root / clock
            folder.mkdir()
            (folder / f"{code}_{clock}.feather").touch()
        (roots[0] / "20260903").mkdir()  # 空目录不构成行情日期。
        (roots[0] / "notes").mkdir()
        calendar = infer_market_calendar(roots,date(2026,9,1),date(2026,9,2))
        self.assertEqual(calendar,(date(2026,9,1),date(2026,9,2),date(2026,9,4)))
        # 保留回测结束之后的日期，仅用于剩余交易日判断。
        with self.assertRaisesRegex(ValueError,"超出行情日期范围"):
            infer_market_calendar(roots,date(2026,9,1),date(2026,9,7))

    def test_calendar_and_inventory(self):
        calendar=self.root/"calendar.csv"
        pd.DataFrame({"date":["2026-08-13","2026-08-14","2026-08-15"],"is_trading_day":[True,True,False]}).to_csv(calendar,index=False)
        self.assertEqual(len(load_calendar(calendar,date(2026,8,13),date(2026,8,15))),2)
        with self.assertRaises(ValueError):
            load_calendar(calendar,date(2026,8,12),date(2026,8,15))
        folder=self.root/"20260813"
        folder.mkdir()
        (folder/"MO2609-C-6600_20260813.feather").touch()
        self.assertEqual(len(inventory(self.root,date(2026,8,13),date(2026,8,14),lambda key:key.startswith("MO"))),1)


if __name__ == "__main__":
    unittest.main()
