import os
base_path = os.path.join(os.environ['BASE_PATH'], 'records')

bn_raw_path = os.path.join(os.environ['BN_DATA_PATH'])


###
TASK_MAPPING = {
    "1010101301": {
        "source": "ashare",
        "period": "1d",
        "cycle": "1h"  ## 持仓1 horizon
    },
    "1000201201": {
        "source": "binance", # 交易所
        "period": "1h", # 预测未来周期
        "cycle": "1h"  ## 持仓1 horizon
    },
    "1020101101": {
        "source": "cffex",
        "period": "1m",
        "cycle": "1h"  ## 持仓1 horizon
    }
}



BN_FUTURES_MAP = {'um': 'futures', 'cm': 'futures', 'spot': 'spot'}