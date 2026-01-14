import os, pdb
import pandas as pd
import numpy as np
from lib.fsr001.base import *
from kdutils.macro2 import *
from sklearn.linear_model import LassoCV
 
def train_model(method, task_id, instruments, period, name):
    
    param_id, use_params = load_params(method=method, 
                instruments=instruments, task_id=task_id,
                period=period, name='lassocv')
    

    valid_coefficients = train_model_coefs(method=method, task_id=task_id,
                    instruments=instruments,
                    period=period,name=name,
                    model_class=LassoCV, 
                    model_params=use_params['model_params'])
    pdb.set_trace()
    analysis_results = analyze_feature(valid_coefficients)
    show_results(analysis_results=analysis_results, name=LassoCV.__name__)
    
    selected_features, selection_info = selectio_feature(
        feature_importance=valid_coefficients, 
        method=use_params['select_params']['method'], 
        **use_params['select_params']['params'])

    save_results(param_id=param_id, use_params=use_params,
            selected_features=selected_features,
            selection_info=selection_info,
            method=method, task_id=task_id, 
            instruments=instruments, 
            period=period)
