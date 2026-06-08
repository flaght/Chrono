from lumina.impulse.fixed import *

def pos_sum(ret, n):
    decay_weight = decay_array(n)
    return np.sum(ret * decay_weight[:, np.newaxis], axis=0)

def calc_umr(values, window):
    factor = pd.DataFrame(map(lambda ret: pos_sum(ret, window),
                              rolling_window(values.values, window)),
                          index=values.index,
                          columns=values.columns)
    return factor