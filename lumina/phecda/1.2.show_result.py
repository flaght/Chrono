import pdb
from ultron.kdutils.file import load_pickle
import pandas as pd
from ultron.ump.metrics.score import WrsmScorer, BaseScorer
from ultron.ump.metrics.metrics_base import MetricsBase

score_tuple_array = load_pickle('score_tuple_array.pkl')

pdb.set_trace()
scorer = WrsmScorer(score_tuple_array)

pdb.set_trace()
scorer.fit_score()

# 实例化WrsmScorer，参数weights，只有第二项为1，其他都是0，
# 代表只考虑投资回报来评分
scorer = WrsmScorer(score_tuple_array, weights=[0, 1, 0, 0])
# 返回排序后的队列
scorer_returns_max = scorer.fit_score()
# 因为是倒序排序，所以index最后一个为最优参数
best_score_tuple_grid = score_tuple_array[scorer_returns_max.index[-1]]

pdb.set_trace()
metrics = MetricsBase.show_general(best_score_tuple_grid.orders_pd, best_score_tuple_grid.action_pd,
                                        best_score_tuple_grid.capital, best_score_tuple_grid.benchmark)