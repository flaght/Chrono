import os

base_path = os.path.join(os.environ['BASE_PATH'], os.environ['RECORD_PATH'])
raw_path = os.path.join(os.environ['RAW_DATA_PATH'])

FUTURES_MAP = {'um': 'futures', 'cm': 'futures', 'spot': 'spot'}




PERIOD_MAPPING = {
    "1d": "1day",
    "1h": "1hour"
}

### 10260201  un  binance
### 11260201  dx  ashare

###
TASK_MAPPING = {
    "1010101301": {
        "source": "ashare",
        "period": "1d",
        "cycle": "1h" ## 持仓1 horizon
    },
    "1000201201": {
        "source": "binance",
        "period": "1h",
        "cycle": "1h" ## 持仓1 horizon
    }
}