import os, pdb
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from lib.fsr001.base import *
from kdutils.macro2 import *
from sklearn.linear_model import LassoCV
 

def train_model(method, task_id, instruments, period, name):
    model_params = {'eps':0.001, 'n_alphas':100, 'cv':5,
                    'positive':True,'max_iter':5000,
                    'n_jobs':-1}
    valid_coefficients = train_model_coefs(method=method, task_id=task_id,
                    instruments=instruments,
                    period=period,name=name,
                    model_class=LassoCV, model_params=model_params)
    
    analysis_results = analyze_feature(valid_coefficients)
    show_results(analysis_results=analysis_results, name=LassoCV.__name__)
    
    selectio_feature(feature_importance=valid_coefficients)