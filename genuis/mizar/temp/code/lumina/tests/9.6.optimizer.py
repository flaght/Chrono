import os, sys, pdb, re, math, json
import sqlalchemy as sa
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath('../'))

import random
from lumina.genetic.fusion.optimizer import mean_variance_builder

# 输入数据
current_positions = np.array([0.25, 0.25, 0.25, 0.25])  # 当前持仓

s1_returns = np.array([0.01, 0.02, 0.04, 0.05])
s2_returns = np.array([0.02, 0.15, 0.03, 0.04])  # 修改为不同序列
s3_returns = np.array([0.05, 0.25, 0.05, 0.45])  # 修改为不同序列
s4_returns = np.array([0.15, 0.1, 0.15, 0.3])  # 修改为不同序列

lower_bound = 0.0  # 权重下限为0
upper_bound = 0.4  # 权重上限为0.5（每个策略最多50%）

all_returns = np.vstack([s1_returns, s2_returns, s3_returns, s4_returns])
er = np.mean(all_returns, axis=1)
cov = np.cov(all_returns)

pdb.set_trace()
### 最大化收益率
status, feval, weights = mean_variance_builder(er=er,
                                               risk_model=cov,
                                               turnover=0.9,
                                               target_vol=0.2,
                                               current_pos=current_positions,
                                               lbound=lower_bound,
                                               ubound=upper_bound)
print("Max Returns Weights:", weights)
print("Max Returns  Weights sum:", np.sum(weights))

current_positions = weights
status, feval, weights = mean_variance_builder(er=er,
                                               risk_model=cov,
                                               turnover=0.9,
                                               target_vol=0.2,
                                               current_pos=current_positions,
                                               lbound=lower_bound,
                                               ubound=upper_bound,
                                               mode="calmar")
print("Max Calmar  Weights:", weights)
print("Max Calmar Weights sum:", np.sum(weights))