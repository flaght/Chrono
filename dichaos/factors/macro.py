from generate import create_factors

basic_factors = [{
    "data1": "大买单成交额/成交额"
}, {
    "data2": "（大买单成交额-大卖单成交额）/成交额"
}, {
    "data3": "大卖单成交额均值/大买单成交额标准差"
}, {
    "data4": "均值（大买单成交额-大卖单成交额）/标准差（大买单成交额-大卖单成交额）"
}, {
    "data5": "大买单成交额/成交额"
}, {
    "data6": "（大买单成交额-大卖单成交额）/成交额"
}, {
    "data7": "大卖单成交额均值/大买单成交额标准差"
}, {
    "data8": "均值（大买单成交额-大卖单成交额）/标准差（大买单成交额-大卖单成交额"
}, {
    "data9": "大卖单成交额/总成交金额"
}, {
    "data10": "（大买成交额+大卖成交金额）/成交额"
}, {
    "data11":
    "高档位的OFI因子选股效果更优，因此第五档的信息含量最高，第一档的信息含量最低，我们提出利用衰减加权的方式对OFI因子进行求和"
}, {
    "data12": "发起市价交易的卖方所付出的交易费用"
}, {
    "data13": "发起市价交易的买方所付出的交易费用"
}, {
    "data14":
    "综合买卖单流动性对于短期收益的影响，可以做𝑀𝐶𝐼𝐴和𝑀𝐶𝐼𝐵的差值来衡量买卖双方交易费用的差值，进而代表价格压力"
}, {
    "data15": "APB 因子分母上的成交量加权价格替换为买单委托量加权的平均价格即可得到订单改进的APB 因子"
}]

expression = [{
    'usage': "MConVariance(5, LAST('factor_1'), LAST('factor_0'))",
    'descri': "时序协方差"
}, {
    'usage': "MRes(5, LAST('factor_0'), LAST('factor_1'))",
    'descri': "滚动残差 取最新值"
}, {
    'usage': "MMeanRes(5, LAST('factor_0'), LAST('factor_1'))",
    'descri': "滚动残差 取均值"
}, {
    'usage': "MCoef(5, LAST('factor_0'), LAST('factor_1'))",
    'descri': "滚动回归系数"
}, {
    'usage': "MRSquared(5, LAST('factor_0'), LAST('factor_1'))",
    'descri': "滚动回归R方"
}, {
    'usage': "EMA(5, LAST('factor_1'))",
    'descri': "指数加权滚动"
}, {
    'usage': " MSUM(5, LAST('factor_1'))",
    "descri": "滚动累计求和"
}]

example_factors = [{
    "factor": "EMA(18,EMA(18,'data0'))",
    "direction": 1,
    "fitness": 0.85
}, {
    "factor": "EMA(10,EMA(10,MSUM(10,MSUM(2,'data1')))))",
    "direction": -1,
    "fitness": 0.75
}, {
    "factor": "EMA(14,EMA(14,EMA(14,'data3')))",
    "direction": 1,
    "fitness": 0.78
}]

basic_data = create_factors(start_date='2022-01-01',
                            end_date='2025-01-01',
                            m=17,
                            n=50,
                            res_name='nxt1_ret')
