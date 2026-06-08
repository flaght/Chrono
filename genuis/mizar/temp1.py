#### 用于notebook 调试

import pandas  as pd
import numpy as np
import pdb, os, datetime, itertools, time, hashlib
from lumina.genetic.util import create_id
from dotenv import load_dotenv

load_dotenv()
from lib.flp001 import *

method = 'bicso2'
task_id = '113001'
instruments = 'rbb'
period = 5

draft_data = load_data2(method=method, task_id=task_id, instruments=instruments, 
          period=period)