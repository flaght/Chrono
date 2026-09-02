import asyncio, os, pdb, json
import random
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from kdutils.macro import base_path
from lib.fes001 import FeatureScorer


def load_results(method, period):
    dir_path = Path(os.path.join(base_path, "prod", method, str(period)))
    snapshot_dict = {}
    for json_file in dir_path.glob("*.json"):
        date_str = json_file.stem.split("_")[0]
        with open(json_file, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
            snapshot_dict[date_str] = snapshot
    return snapshot_dict


async def run(method, period):
    ticker = "000852"
    snapshot_dict = await asyncio.to_thread(load_results,
                                            method=method,
                                            period=period)
    evaluator = FeatureScorer(min_samples=2, min_win_rate=0.50, min_contribution=0.0)
    for k, v in snapshot_dict.items():
        # v['forward_return'] = random.uniform(-1, 1)
        # v["status"] = "SETTLED"
        evaluator.process_record(record_data=v)
        
    res = evaluator.evaluate_generate()
  
    print("📊 【三模态统一特征统计全景报告】:")
    print("-" * 90)
    print(f"{'模态层级':<14} {'特征名称/类别':<28} {'状态':<24} {'引用':<6} {'胜率':<8} {'累计贡献分':<12}")
    print("-" * 90)
    for row in res["detailed_reports"]:
        print(f"{row['layer']:<14} {row['display_name']:<28} {row['status']:<24} {row['referenced_count']:<6} {row['win_rate']*100:.1f}%   {row['cumulative_contribution']:+.6f}")

    print("-" * 90)
    print("\n🌟 【三模态分层输出结果】:")
    print(f"1. PREDICT (预测特征白名单):\n   - active: {res['PREDICT']['active']}\n   - insufficient: {res['PREDICT']['insufficient_evidence']}")
    print(f"2. REGIME (市场环境特征):\n   - active: {res['REGIME']['active']}\n   - insufficient: {res['REGIME']['insufficient_evidence']}")
    print(f"3. TEXTUAL (稳定文本类别):\n   - active: {res['TEXTUAL']['active']}\n   - insufficient: {res['TEXTUAL']['insufficient_evidence']}")
    print("=" * 90)


if __name__ == '__main__':
    method = 'test0'
    period = 3
    asyncio.run(run(method=method, period=3))
