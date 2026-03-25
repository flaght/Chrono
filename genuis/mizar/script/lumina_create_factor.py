#### 直接使用原始特征
### 输出 trade_time code + 特征+  nxt1_ret_15h 收益率
from dotenv import load_dotenv
import pdb,os
load_dotenv()
from lib.aux001 import fetch_market
from lib.svx001 import scale_factors
from kdutils.macro import *

method = 'dicso2'
instruments = 'rbb'
task_id = '113001'
datasets = ['train','val','test']
period = 15
factors_data, returns_data = fetch_market(method=method,
                                          instruments=instruments,
                                          task_id=task_id,datasets=datasets)
factor_columns = [col for col in factors_data.columns if col not in ['trade_time','code']]
### 标准化
pdb.set_trace()
#factor_columns = factor_columns[0:10]
for col in factor_columns:
    print(col)
    scale_factors(predict_data=factors_data,
                      method='roll_zscore',
                      win=15,
                      factor_name=col)
    factors_data[col] = factors_data['transformed']
    factors_data.drop(['transformed'], axis=1, inplace=True)
total_data = factors_data[['trade_time','code'] + factor_columns].merge(returns_data[['trade_time','code', "nxt1_ret_{}h".format(period)]])
output_dirs = os.path.join(base_path, method, instruments, "temp", "model",str(task_id), str(period))
if not os.path.exists(output_dirs):
    os.makedirs(output_dirs)

filename = os.path.join(output_dirs, "final_all_data.feather")
total_data.to_feather(filename)