## 选中因子生成绩效文件
import itertools, os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from kdutils.tactix import Tactix
from kdutils.macro2 import *
from lib.smx001 import create_metrics


if __name__ == '__main__':
    variant = Tactix().start()
    create_metrics(method=variant.method,
         instruments=variant.instruments,
         period=variant.period,
         task_id=variant.task_id)
    
