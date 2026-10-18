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


# 全市场平均成本(ASR)与最集中成本(Peak)的偏离。若ASR高于Peak，可能说明近期有资金在高位建仓。
def cp020(chip_data, close):
    asr = np.average(chip_data['price'], weights=chip_data['chips'])
    peak_idx = chip_data['chips'].idxmax()
    chip_peak_price = chip_data.loc[peak_idx, 'price']
    alpha = (asr - chip_peak_price) / (asr + 1e-9)
    alpha = pd.DataFrame(np.array([alpha]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
