import os

base_path = os.path.join(os.environ['BASE_PATH'], os.environ['RECORD_PATH'])
raw_path = os.path.join(os.environ['RAW_DATA_PATH'])

FUTURES_MAP = {'um': 'futures', 'cm': 'futures', 'spot': 'spot'}
