import numpy as np
import pandas as pd
import pdb

score_pd = pd.read_feather('score_pd.feather')

columns_name = score_pd.columns

score_ls = np.linspace(0, 1, score_pd.shape[0])
pdb.set_trace()
for cn in columns_name:
    print(cn)
    score = score_ls[(score_pd[cn].fillna(0).rank().values -1).astype(int)]
