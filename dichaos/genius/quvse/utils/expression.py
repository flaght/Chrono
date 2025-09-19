
expression_str_md = """

| Category | Expression | Name | Description|
| :--- | :--- | :--- | :--- |
| **基础算子** | `ADDED(x, y)` | 加法 (Add) | 因子x和y逐元素相加。 |
| **基础算子** | `SUBBED(x, y)` | 减法 (Subtract) | 因子x和y逐元素相减。 |
| **基础算子** | `MUL(x, y)` | 乘法 (Multiply) | 因子x和y逐元素相乘。 |
| **基础算子** | `DIV(x, y)` | 除法 (Divide) | 因子x和y逐元素相除。 |
| **基础算子** | `MOD(x, y)` | 求模 (Modulo) | 因子x对y逐元素求模。 |
| **基础算子** | `POW(x, n)` | 幂运算 (Power) | 计算因子x的n次方。 |
| **基础算子** | `MINIMUM(x, y)` | 较小值 (Minimum) | 逐元素返回x和y中较小的值。 |
| **基础算子** | `MAXIMUM(x, y)` | 较大值 (Maximum) | 逐元素返回x和y中较大的值。 |
| **基础算子** | `ABS(x)` | 绝对值 (Absolute) | 计算因子x的绝对值。 |
| **基础算子** | `SIGN(x)` | 符号函数 (Sign) | 返回因子x的符号（-1, 0, 1）。 |
| **基础算子** | `LOG(x)` | 自然对数 (Natural Log) | 计算因子x的自然对数。 |
| **基础算子** | `LOG2(x)` | Log2对数 | 计算以2为底的对数。 |
| **基础算子** | `LOG10(x)` | Log10对数 | 计算以10为底的对数。 |
| **基础算子** | `EXP(x)` | 指数函数 (Exponential) | 计算e的x次方。 |
| **基础算子** | `SQRT(x)` | 平方根 (Square Root) | 计算因子x的平方根。 |
| **基础算子** | `ACOS(x)` | 反余弦 (Arc Cosine) | 计算因子x的反余弦。 |
| **基础算子** | `ASIN(x)` | 反正弦 (Arc Sine) | 计算因子x的反正弦。 |
| **基础算子** | `TANH(x)` | 双曲正切 (Hyperbolic Tan) | 计算因子x的双曲正切。 |
| **基础算子** | `CEIL(x)` | 向上取整 (Ceiling) | 对因子x向上取整。 |
| **基础算子** | `FLOOR(x)` | 向下取整 (Floor) | 对因子x向下取整。 |
| **基础算子** | `ROUND(x)` | 四舍五入 (Round) | 对因子x进行四舍五入。 |
| **基础算子** | `FRAC(x)` | 小数部分 (Fractional) | 返回因子x的小数部分 (`abs(x) - floor(x)`)。 |
| **基础算子** | `SIGMOID(x)` | Sigmoid函数 | 计算 `1 / (1 + exp(-x))`。 |
| **基础算子** | `RELU(x)` | ReLU激活函数 | 计算 `max(0, x)`。 |
| **基础算子** | `NORMINV(x)` | 正态分布逆函数 | 标准正态分布的逆累积分布函数。 |
| **基础算子** | `SIGLOGABS(x)` | 符号对数 | 计算 `sign(x) * log(abs(x) + 1)`。 |
| **基础算子** | `SIGLOG2ABS(x)` | 符号对数 (Base 2) | 计算 `sign(x) * log2(abs(x) + 1)`。 |
| **基础算子** | `SIGLOG10ABS(x)`| 符号对数 (Base 10)| 计算 `sign(x) * log10(abs(x) + 1)`。 |
| **基础算子** | `AVG(x)` | 截面均值 | 计算当前截面上所有样本的均值。 |
| **基础算子** | `DIFF(x)` | 一阶差分 | 计算当前值与上一个周期值的差，等价于 `DELTA(1, x)`。 |
| **时序算子** | `MA(window, x)` | 滚动均值 | 计算因子x在过去`window`个周期内的移动平均值。 |
| **时序算子** | `MSTD(window, x)` | 移动标准差 | 计算因子x在过去`window`个周期内的移动标准差。 |
| **时序算子** | `MVARIANCE(window, x)` | 时序方差 | 计算因子x在过去`window`个周期内的移动方差。 |
| **时序算子** | `MSUM(window, x)` | 滚动求和 | 计算因子x在过去`window`个周期内的累加值。 |
| **时序算子** | `MPRO(window, x)` | 滚动累乘 | 计算因子x在过去`window`个周期内的累乘值。 |
| **时序算子** | `MMAX(window, x)` | 周期最大值 | 获取因子x在过去`window`个周期内的最大值。 |
| **时序算子** | `MMIN(window, x)` | 周期最小值 | 获取因子x在过去`window`个周期内的最小值。 |
| **时序算子** | `MMedian(window, x)` | 时序中位数 | 计算因子x在过去`window`个周期内的中位数。 |
| **时序算子** | `MARGMAX(window, x)` | 周期最大值位序 | 获取因子x在过去`window`个周期内最大值的位置索引。 |
| **时序算子** | `MARGMIN(window, x)` | 周期最小值位序 | 获取因子x在过去`window`个周期内最小值的位置索引。 |
| **时序算子** | `MRANK(window, x)` | 时序排序 | 计算当前值在过去`window`个周期内的排序（从小到大）。 |
| **时序算子** | `MQUANTILE(window, x)` | 时序分位数 | 计算当前值在过去`window`个周期内的分位数。 |
| **时序算子** | `MPERCENT(window, x)` | 时序百分位 | 计算当前值在过去`window`个周期内的百分位排名。 |
| **时序算子** | `MSKEW(window, x)` | 移动偏度 | 计算因子x在过去`window`个周期内的偏度。 |
| **时序算子** | `MKURT(window, x)` | 移动峰度 | 计算因子x在过去`window`个周期内的峰度。 |
| **时序算子** | `MCORR(window, x, y)` | 滚动相关性 | 计算因子x和y在过去`window`个周期内的相关系数。 |
| **时序算子** | `MConVariance(window, x, y)`| 时序协方差 | 计算因子x和y在过去`window`个周期内的协方差。 |
| **时序算子** | `MCoef(window, x, y)` | 滚动回归系数 | 计算在过去`window`个周期内，y对x的回归系数(beta)。 |
| **时序算子** | `MRes(window, x, y)` | 滚动残差 | 计算在过去`window`个周期内，y对x的回归残差的最新值。 |
| **时序算子** | `MMeanRes(window, x, y)`| 滚动平均残差 | 计算在过去`window`个周期内，y对x的回归残差的平均值。 |
| **时序算子** | `MRSquared(window, x, y)`| 滚动回归R方 | 计算在过去`window`个周期内，y对x的回归R²值。 |
| **时序算子** | `SHIFT(window, x)` | 向前取值 | 获取因子x在`window`个周期前的值。 |
| **时序算子** | `DELTA(window, x)` | 周期差值 | 计算因子x当前值与`window`个周期前的值的差。 |
| **时序算子** | `EMA(window, x)` | 指数移动平均 | 计算因子x的指数移动平均值。 |
| **时序算子** | `WMA(window, x)` | 加权移动平均 | 计算因子x的线性加权移动平均值。 |
| **时序算子** | `MDEMA(window, x)` | 双重移动均线 | 计算因子x的双重指数移动平均值，减少延迟。 |
| **时序算子** | `MT3(window, x)` | 三指数移动均线 | 计算因子x的三重指数移动平均值(T3)，更平滑。 |
| **时序算子** | `MHMA(window, x)` | 变色移动均线 | 计算Hull移动平均线，响应快且平滑。 |
| **时序算子** | `MACD(window_fast, window_slow, x)` | 异同移动平均线 | 计算因子x的MACD指标。 |
| **时序算子** | `RSI(window, x)` | 相对强弱指数 | 计算因子x在过去`window`个周期内的相对强弱指数。 |
| **时序算子** | `MADiff(window, x)` | 偏离均值 | 计算当前值与滚动均值的差值 `x - MA(window, x)`。 |
| **时序算子** | `MADecay(window, x)` | 滚动衰退值 | 计算因子x在过去`window`个周期内的线性衰减加权平均值。|
| **时序算子** | `MACCBands(window, x, y)` | ACCBands通道 | 两个不同周期线值相减，通常用于构建通道。 |
| **时序算子** | `MMASSI(window, x, y)` | 梅斯线 (Mass Index) | 基于高低价范围的波动性指标，用于预测趋势反转。 |
| **时序算子** | `MPWMA(window, x, y)` | WMA商 | 计算两个因子滚动加权平均值的商。 |
| **时序算子** | `MIChimoku(window, x, y)` | 滚动IChimoku指标 | 计算一目均衡表(Ichimoku)中的部分指标。 |
| **时序算子** | `MSLMean(window, x, y)` | 时间切割均值 | 对窗口进行切割比较，例如前半段均值与后半段均值的关系。|
| **时序算子** | `MSmart(window, x, y)` | 滚动聪明指标 | 一种加权的滚动指标。 |
| **时序算子** | `MCPS(window, x)` | 滚动范围压缩 | 计算 `(min + max) - 0.5 * x`。 |
| **时序算子** | `MDIFF(window, x)` | 滚动中心化 | 计算 `x - 0.5 * (min + max)`。 |
| **时序算子** | `MMaxDiff(window, x)` | 与最大值差值 | 计算 `x - MMAX(window, x)`。 |
| **时序算子** | `MMinDiff(window, x)` | 与最小值差值 | 计算 `x - MMIN(window, x)`。 |
| **时序算子** | `MVHF(window, x)` | 十字过滤 (VHF) | 垂直水平过滤指标，用于判断市场处于趋势还是盘整。 |
| **时序算子** | `MDPO(window, x)` | 区间震荡线 (DPO) | 消除价格趋势，识别价格周期。 |
| **时序算子** | `MIR(window, x)` | 信息比率 | 类似信息比率的计算，双重均值除以标准差。 |
| **时序算子** | `MALLTRUE(window, x)` | 滚动全为正 | 判断过去`window`周期内是否所有值都为正。 |
| **时序算子** | `MANYTRUE(window, x)` | 滚动存在正 | 判断过去`window`周期内是否存在正值。 |
| **时序算子** | `MNPOSITIVE(window, x)` | 滚动正数统计 | 统计过去`window`周期内正数的个数。 |
| **时序算子** | `MAPOSITIVE(window, x)` | 滚动正数均值 | 计算过去`window`周期内所有正数的均值。 |
"""