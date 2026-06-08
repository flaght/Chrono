from scipy.signal import find_peaks
import pandas as pd
import numpy as np
'''
chips_data:
        price     chips
7     8.12535  1.642764
12    8.25060  0.113686
13    8.27565  0.174279
15    8.32575  1.801554
'''


# 筹码分布中显著峰值的数量。单峰密集为佳；多峰形态意味着存在多个套牢盘，上行阻力重重。
def cp005(chip_data, close, chip_peak_strength, distance=5):
    peaks, _ = find_peaks(chip_data['chips'],
                          height=chip_peak_strength * 0.2,
                          distance=distance)
    alpha1 = len(peaks)
    alpha = pd.DataFrame(np.array([alpha1]),
                         index=close.index[-1:],
                         columns=close.columns)
    return alpha
