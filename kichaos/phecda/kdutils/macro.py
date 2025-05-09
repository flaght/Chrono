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

base_path = os.path.join('/workspace/data/data/basic_data/dx_data/',
                         os.environ['KICHAOS_BASE_NAME'])

codes = instruments_codes[os.environ['INSTRUMENTS']]

### 0 sklearn standard
### 1 -1~1 等比例扩充
### 2 rolling 标准化

normal_categories = {
    'rolling': 2,
}

### 合约连乘数
CONT_MULTNUM_MAPPING = {'RB': 10, 'IF': 300, 'IM': 200}

### 手续费
COST_MAPPING = {
    'RB': {
        'buy': 0.00012,
        'sell': 0.00012
    },
    'IF': {
        'buy': 0.000023,
        'sell': 0.000023
    },
    'IM':{
        'buy': 0.000023,
        'sell': 0.000023
    }
}

### 初始资金
INIT_CASH_MAPPING = {
    'RB': 60000.0,
    'IF': 2000000.0,
    'IM': 2000000.0

}

### 平仓时间
CLOSE_TIME_MAPPING = {
    'IF': [('14:58:00', '15:00:00')],
    'IM': [('14:58:00', '15:00:00')],
    'RB':[('22:58:00', '23:00:00'), ('14:58:00', '15:00:00')]
}

#### 
THRESHOLD_MAPPING = {
    'RB': {
        'long_open':0.55,#0.52,#0.55,
        'long_close':0.6,#0.58,#0.6,
        'short_open':0.52,
        'short_close':0.58
    },
    'IF': {
        'long_open':0.55,
        'long_close':0.6,
        'short_open':0.55,
        'short_close':0.6
    },
    'IM': {
         'long_open':0.58,
        'long_close':0.62,
        'short_open':0.58,
        'short_close':0.62
    }
}