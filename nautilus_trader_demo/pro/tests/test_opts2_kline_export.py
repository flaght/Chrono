"""无需DDB服务，验证期货导出字段与累计口径。"""
from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from examples.opts2.kline_task import (FUTURE_COLUMNS, load_future_specs,
    prepare_bars, prepare_future_bars, export_day, infer_future_specs)


class FutureExportTests(unittest.TestCase):
    def frame(self, code="IM2609"):
        return pd.DataFrame(dict(Code=[code]*3,symbol=[code]*3,
            date=[pd.Timestamp("2026-09-11")]*3,
            minTime=pd.to_datetime(["1970-01-01 09:30","1970-01-01 09:31","1970-01-01 09:32"]),
            open=[7000.]*3,high=[7100.]*3,low=[6900.]*3,close=[7000.]*3,
            volume=[0,2,5],turnover=[0.,2800000.,7040000.],open_interest=[50.,51.,49.]))

    def test_exact_legacy_schema_and_placeholder_vwap(self):
        spec=dict(symbol="IM2609",exchange="CCFX",multiplier=200.)
        bars=prepare_future_bars(self.frame(),date(2026,9,11),"Asia/Shanghai","error",spec)
        self.assertEqual(tuple(bars.columns),FUTURE_COLUMNS)
        self.assertEqual(bars.volume.tolist(),[0.,2.,3.])
        self.assertEqual(bars.value.tolist(),[0.,2800000.,4240000.])
        np.testing.assert_allclose(bars.vwap,bars.close)
        self.assertEqual(bars.vt_symbol.iloc[0],"IM2609.CCFX")
        self.assertEqual(bars.date.iloc[0],"2026-09-11")
        self.assertEqual(bars.time.iloc[0],"09:30:00")
        self.assertIsNone(bars.datetime.dt.tz)
        self.assertEqual(bars.datetime.iloc[0],pd.Timestamp("2026-09-11 09:30"))

    def test_partial_first_row_and_missing_interest_rejected(self):
        spec=dict(symbol="IM2609",exchange="CCFX",multiplier=200.)
        frame=self.frame().iloc[1:].copy()
        with self.assertRaisesRegex(ValueError,"首条累计值"):
            prepare_future_bars(frame,date(2026,9,11),"Asia/Shanghai","error",spec)
        with self.assertRaisesRegex(ValueError,"open_interest"):
            prepare_future_bars(self.frame().drop(columns="open_interest"),date(2026,9,11),"Asia/Shanghai","error",spec)

    def test_metadata_preserves_contract_case_and_exchange(self):
        basic=pd.DataFrame([dict(symbol="rb2703",exchangeCD="XSGE",contMultNum=10),
                            dict(symbol="IM2609",exchangeCD="CCFX",contMultNum=200)])
        with patch("examples.opts2.kline_task.pd.read_feather",return_value=basic):
            specs=load_future_specs("fut_basic.feather")
        self.assertEqual(specs["RB2703"],dict(symbol="rb2703",exchange="XSGE",multiplier=10.))
        self.assertEqual(specs["IM2609"]["multiplier"],200.)

    def test_mixed_table_exports_only_futures_and_correct_filename(self):
        class Session:
            def upload(self,values):
                self.values=values
            def run(self,sql):
                self.sql=sql
                return pd.concat([self_frame,self_frame.assign(Code="MO2609-C-7000",symbol="MO2609-C-7000")])
        self_frame=self.frame("im2609")
        session=Session()
        written=[]
        def write(frame,path,**kwargs):
            written.append(frame.copy())
            Path(path).touch()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with patch.object(pd.DataFrame,"to_feather",write):
                files,rows=export_day(session,"dfs://min_bar","cffex_1min",date(2026,9,11),
                    "future","Asia/Shanghai","error",directory,
                    future_specs={"IM2609":dict(symbol="IM2609",exchange="CCFX",multiplier=200.)})
            self.assertTrue((Path(directory)/"20260911"/"IM2609_20260911.feather").is_file())
        self.assertEqual((files,rows),(1,3))
        self.assertEqual(tuple(written[0].columns),FUTURE_COLUMNS)
        self.assertIn("where date = 2026.09.11",session.sql)
        self.assertNotIn("product",session.sql)

    def test_without_basic_exports_cffex_and_zero_vwap(self):
        source=pd.concat([self.frame(),self.frame("SF702").assign(turnover=[0.,100.,90.])])
        class Session:
            def upload(self,values): pass
            def run(self,sql): return source
        written=[]
        def write(frame,path,**kwargs):
            written.append(frame.copy())
            Path(path).touch()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            with patch.object(pd.DataFrame,"to_feather",write):
                counts=export_day(Session(),"dfs://min_bar","cffex_1min",date(2026,9,11),
                    "future","Asia/Shanghai","error",directory)
            self.assertTrue((Path(directory)/"20260911"/"IM2609_20260911.feather").is_file())
        self.assertEqual(counts,(1,3))
        self.assertEqual(tuple(written[0].columns),FUTURE_COLUMNS)
        self.assertEqual(written[0].vt_symbol.tolist(),["IM2609.CCFX"]*3)
        self.assertEqual(written[0].vwap.tolist(),[0.0]*3)
        self.assertEqual(written[0].volume.tolist(),[0.0,2.0,3.0])

    def test_without_basic_uses_source_exchange_and_rejects_conflicts(self):
        source=self.frame("rb2703.XSGE")
        with self.assertRaisesRegex(ValueError,"缺少交易所"):
            infer_future_specs(self.frame("SF702"),"cffex_1min")
        specs=infer_future_specs(source,"cffex_1min")
        self.assertEqual(specs["RB2703"]["exchange"],"XSGE")
        with self.assertRaisesRegex(ValueError,"冲突"):
            infer_future_specs(source.assign(exchangeCD="CCFX"),"cffex_1min")

    def test_option_schema_remains_unchanged(self):
        bars=prepare_bars(self.frame("MO2609-C-7000"),date(2026,9,11),"Asia/Shanghai","error")
        self.assertIn("trade_date",bars)
        self.assertIn("turnover_accumulate",bars)
        self.assertNotIn("vt_symbol",bars)
        self.assertEqual(str(bars.datetime.dt.tz),"Asia/Shanghai")


if __name__ == "__main__":
    unittest.main()
