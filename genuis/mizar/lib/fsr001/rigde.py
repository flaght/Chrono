import os, pdb
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from lib.fsr001.base import train_model as train_base_model
from kdutils.macro2 import *

def train_model(method, task_id, instruments, period, name):
    model_params = {'alpha':1.0}
    train_base_model(method=method, task_id=task_id,
                    instruments=instruments,
                    period=period,name=name,
                    model_class=Ridge, model_params=model_params)
