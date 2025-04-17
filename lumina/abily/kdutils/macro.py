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
COST_MAPPING = {'IF': 0.000023}

### 滑点
SLIPPAGE_MAPPING = {'IF': 0.0001}
