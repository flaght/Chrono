import os

base_path = os.path.join(os.environ['BASE_PATH'], os.environ['RECORD_PATH'])
raw_path = os.path.join(os.environ['RAW_DATA_PATH'])

FUTURES_MAP = {'um': 'futures', 'cm': 'futures', 'spot': 'spot'}




PERIOD_MAPPING = {
    "1d": "1day",
    "1h": "1hour"
}

### 20260201  dx  ashare
TASK_MAPPING = {
    "20260201": {
        "source": "ashare",
        "period": "1d",
        "cycle": "1h" ## 持仓1 horizon
    },
    "202060101": {
        "source": "binance",
        "period": "1h",
        "cycle": "1h" ## 持仓1 horizon
    }
}