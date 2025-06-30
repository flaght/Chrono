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
COST_MAPPING = {'IF': 0.00023, 'IM': 0.0008}

### 滑点 # 价格在 6000 一次开仓滑0.1 一次平仓滑0.1 共0.2点 滑点百分比 0.2 / 7000 ≈ 0.000028
SLIPPAGE_MAPPING = {'IF': 0.0001, 'IM': 1.7e-05}  # 平均用 0.1 / 6000 

INDEX_MAPPING = {"IF": 100001, 'IM': 100013}  

## IM 100002 卡玛  100012 夏普
## 对应的绩效值
PERFORMANCE_MAPPING = {100002: "profit_calmar", 100012: "profit_sharpe", 100013: "order_std"}
