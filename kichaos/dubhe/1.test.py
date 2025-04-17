import sys, os, pdb, argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
sys.path.append('../../kichaos')

from kdutils.data import *

begin_date = '2018-01-01'
end_date = '2024-10-29'
remain_data = fetch_f1r_oo(begin_date, end_date, 'hs300')
pdb.set_trace()
print(remain_data.head())