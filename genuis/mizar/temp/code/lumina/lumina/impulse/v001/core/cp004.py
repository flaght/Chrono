import pandas as pd
from lumina.impulse.fixed import *
'''
chips_data:
        price     chips
7     8.12535  1.642764
12    8.25060  0.113686
13    8.27565  0.174279
15    8.32575  1.801554
'''


## 覆盖中间80%筹码的价格区间宽度。带宽越窄，市场成本越一致，趋势一旦形成可能越猛烈。
def cp004(chip_data, close, percent=0.1):
    lowper = percent
    upper = 1 - lowper
    total_chips = chip_data['chips'].sum()
    cumulative_chips = chip_data['chips'].cumsum() / total_chips
    lower_idx = chip_data[cumulative_chips > lowper].idxmin() ## 不会为空
    if chip_data[cumulative_chips < upper].empty:
        alpha1 = 0.0
    else:
        upper_idx = chip_data[cumulative_chips < upper].idxmax() 
        lower_price = chip_data.loc[lower_idx.values[0], 'price']
        upper_price = chip_data.loc[upper_idx.values[0], 'price']
        alpha1 =  upper_price - lower_price
    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
