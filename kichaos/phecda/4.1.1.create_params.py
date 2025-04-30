import os, pickle, pdb, argparse, hashlib, json
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
'''
1. 通过预测选择对应的模型环境相关参数
2. 读取依赖的挖掘因子对应表达式
'''

from kdutils.macro import *
from kdutils.util import create_id


def create_params(variant):
    filename = os.path.join(
        "records", "env_params",
        "{0}_{1}_{2}_trader.pkl".format(variant['code'], variant['direction'],
                                        variant['environment']))

    pdb.set_trace()
    with open(filename, "rb") as f:
        params = pickle.load(f)

    dirs = os.path.join(base_path, variant['method'], 'fitness',
                        variant['g_instruments'], 'evolution')
    filename = os.path.join(
        dirs, '{0}_{1}h.feather'.format(variant['categories'],
                                        variant['horizon']))
    formual_data = pd.read_feather(filename)
    formual_data = formual_data.set_index('name').loc[params['features']].reset_index()

    params['formual'] = formual_data.drop(['update_time'],
                                          axis=1).to_dict(orient='records')

    ## 创建
    s = hashlib.md5(json.dumps(params).encode(encoding="utf-8")).hexdigest()
    task_id = create_id(original=s, digit=10)
    params['task_id'] = task_id

    base_path1 = os.path.join("records", "model_params")
    if not os.path.exists(base_path1):
        os.makedirs(base_path1)
    filename1 = os.path.join(
        base_path1,
        "{0}_{1}_{2}_{3}_trader.pkl".format(variant['code'], variant['direction'],
                                        variant['environment'], task_id))
    
    with open(filename1, "wb") as f:
        pickle.dump(params, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--code', type=str, default='IM')
    parser.add_argument('--g_instruments', type=str, default='ims')
    parser.add_argument('--direction', type=str, default='long')
    parser.add_argument('--environment', type=str, default='hedge041')
    parser.add_argument('--method', type=str, default='aicso2')
    parser.add_argument('--categories', type=str, default='o2o')
    parser.add_argument('--horizon', type=str, default='1')

    args = parser.parse_args()

    create_params(vars(args))
