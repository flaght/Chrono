import os

## brief

#codes = ['BU', 'FU', 'HC', 'L', 'M', 'MA', 'PP', 'RB', 'SR', 'TA', 'V']

instruments_codes = {
    'brief': ['BU', 'FU', 'HC', 'L', 'M', 'MA', 'PP', 'RB', 'SR', 'TA', 'V'],
    'istocks': ['IF', 'IM', 'IC', 'IH'],
    'ifs': ['IF'],
    'ihs': ['IH'],
    'ics': ['IC'],
    'ims': ['IM'],
    'rbb': ['RB']
}

base_path = os.path.join('./', 'records')

codes = instruments_codes[os.environ['INSTRUMENTS']]

### 合约连乘数
CONT_MULTNUM_MAPPING = {'RB': 10, 'IF': 300, 'IM': 200}

### 手续费
COST_MAPPING = {'IF': 0.000023, 'IM': 0.000023}

### 滑点
SLIPPAGE_MAPPING = {'IF': 0.0001, 'IM': 0.0001}

INDEX_MAPPING = {"IF": 100001, 'IM': 100012}  

## IM 100002 卡玛  100012 夏普
## 对应的绩效值
PERFORMANCE_MAPPING = {100002: "calmar", 100012: "sharpe"}
