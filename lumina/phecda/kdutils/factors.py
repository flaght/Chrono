import numpy as np
from lumina.factors.buy import MinFactorKingKeltnerBuyL, MinFactorKingKeltnerBuyS
from lumina.factors.sell import MinFactorKingKeltnerSell
from lumina.factors.buy import MinFactorBollBuyL, MinFactorBollBuyS
from lumina.factors.sell import MinFactorBollSell
from lumina.factors.buy import MinFactorChaosBuyL, MinFactorChaosBuyS
from lumina.factors.sell import MinFactorChaosSell
from lumina.factors.sell import MinFactorAtrNStop
from lumina.factors.sell import MinFactorCloseAtrNStop
from lumina.position.position_fake import FakePosition

factor_risk = [{
    'class': [MinFactorAtrNStop],
    'stop_loss_n': np.arange(1.5, 2.5, 1),
    'stop_win_n': np.arange(0.5, 1.5, 1)
}, {
    'class': [MinFactorCloseAtrNStop],
    'close_atr_n': [2.5]
}]
factors_sell = [{
    'class': [MinFactorKingKeltnerSell],
    'ma_xd': [5, 10],
    'ewm': [0],
}, {
    'class': [MinFactorChaosSell],
    'fast': [3],
    'slow': [10],
}, {
    'class': [MinFactorBollSell],
    'ma_xd': [10, 15],
    'offset': [1.5],
    'ewm': [0],
    'roc': [5]
}]

factors_buy = [{
    'class': [MinFactorKingKeltnerBuyL],
    'ewm': [0],
    'change_threshold': [0.12],
    'position': [{
        'class': FakePosition
    }]
}, {
    'class': [MinFactorKingKeltnerBuyS],
    'ewm': [1],
    'change_threshold': [0.12],
    'position': [{
        'class': FakePosition
    }]
}, {
    'class': [MinFactorChaosBuyL],
    'fast': [5],
    'slow': [10],
    'ewm': [0, 1],
    'position': [{
        'class': FakePosition
    }]
}, {
    'class': [MinFactorChaosBuyS],
    'fast': [5],
    'slow': [10],
    'ewm': [0, 1],
    'position': [{
        'class': FakePosition
    }]
}, {
    'class': [MinFactorBollBuyL],
    'ma_xd': [10],
    'ewm': [0, 1],
    'change_threshold': [0.12],
    'roc': [5],
    'offset': [1.5],
    'position': [{
        'class': FakePosition
    }]
}, {
    'class': [MinFactorBollBuyS],
    'ma_xd': [10],
    'ewm': [0, 1],
    'change_threshold': [0.12],
    'roc': [3],
    'offset': [1.5],
    'position': [{
        'class': FakePosition
    }]
}]

factors_opertors = {
    'buy': factors_buy,
    'sell': factors_sell,
    'risk': factor_risk
}

factors_sets = {}
factors_sets['rbb'] = factors_opertors
