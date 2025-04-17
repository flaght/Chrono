import numpy as np
#from lumina.factors.buy import MinFacotorBaMaBuyL, MinFacotorBaMaBuyS
from lumina.factors.sell import MinFactorAtrNStop
from lumina.factors.sell import MinFactorCloseAtrNStop
from lumina.factors.sell import MinFactorPreAtrNStop
from lumina.factors.buy import MinFactorChaosBuyL, MinFactorChaosBuyS
from lumina.factors.sell import MinFactorChaosSell
from lumina.factors.buy import MinFactorKingKeltnerBuyL, MinFactorKingKeltnerBuyS
from lumina.factors.sell import MinFactorKingKeltnerSell
from lumina.factors.buy import MinFactorTiChaosBuyL
from lumina.factors.sell import MinFactorTiChaosSell
from lumina.factors.buy import MinFactorBollBuyL, MinFactorBollBuyS
from lumina.factors.sell import MinFactorBollSell
from lumina.factors.buy import MinFactorRSRSBuyL, MinFactorRSRSBuyS
from lumina.factors.sell import MinFactorRSRSSell
from lumina.position.position_fake import FakePosition


factors_temp1 = {
    'buy': [{
        'class': [MinFactorKingKeltnerBuyL],
        'cycle': [10],
        'ewm': [0, 1],
        'position': [{
            'class': FakePosition
        }]
    }, {
        'class': [MinFactorKingKeltnerBuyS],
        'cycle': [10],
        'ewm': [0, 1],
        'position': [{
            'class': FakePosition
        }]
    }, {
        'class': [MinFactorChaosBuyL],
        'cycle': [10],
        'ewm': [0, 1],
        'position': [{
            'class': FakePosition
        }]
    }, {
        'class': [MinFactorChaosBuyS],
        'cycle': [10],
        'ewm': [1],
        'position': [{
            'class': FakePosition
        }]
    }],
    'sell': [{
        'class': [MinFactorKingKeltnerSell],
        'ma_xd': [10],
        'atr_xd': [5],
        'ewm': [0, 1],
    }, {
        'class': [MinFactorChaosSell],
        'fast': [3],
        'slow': [5, 10],
    }],
    'risk': [{
        'class': [MinFactorAtrNStop],
        'stop_loss_n': np.arange(2.0, 2.5, 0.5),
        'stop_win_n': np.arange(0.5, 1., 0.5)
    }]
}
factors_temp2 = {
    'buy': [{
        'class': [MinFactorTiChaosBuyL],
        'threshold': [0.02],
        'position': [{
            'class': FakePosition
        }]
    }],
    'sell': [{
        'class': [MinFactorTiChaosSell],
        'threshold': [-0.03]
    }]
}
factors_temp3 = {
    'buy': [{
        'class': [MinFactorBollBuyL],
        'resample_min': [3],
        'resample_max': [20],
        'ewm': [0],
        'roc': [30],
        'position': [{
            'class': FakePosition
        }]
    },{
        'class': [MinFactorBollBuyS],
        'resample_min': [3],
        'resample_max': [20],
        'ewm': [0],
        'roc': [30],
        'position': [{
            'class': FakePosition
        }]
    }],
    'sell': [{
        'class': [MinFactorBollSell],
        'ewm': [0, 1],
    }]
}

factors_temp4 = {
    'buy': [{
        'class': [MinFactorRSRSBuyL],
        'resample_min': [3],
        'resample_max': [20],
        'ewm': [0],
        'position': [{
            'class': FakePosition
        }]
    }],
    'risk': [{
        'class': [MinFactorCloseAtrNStop],
        'close_atr_n': [2],#np.arange(2.0, 2.5, 0.5)
    }]
}

factors_sets = {}

factors_sets['ims'] = factors_temp4
factors_sets['rbb'] = factors_temp4
