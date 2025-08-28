import pdb, os
from typing import Dict, Any, Optional, Tuple, List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from dichaos.services.react.tool import BaseTool, ToolResult
from dichaos.services.llm import LLMServiceFactory, MessageGenerator
from dichaos.env import DEBUG_MODE

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

EXPS_SYSTEM_PROMPT = """
你是一位顶级的**量化因子构建专家**，不仅精通数学表达式，更是一位富有创造力的策略师。你的工作是基于一个核心思路，从**多个不同维度**进行发散性思考，构建出一系列**结构各异、逻辑多样**的因子。

你的核心任务是：根据用户提供的单一因子思路，**创造性地生成10个本质不同**的因子变体，并对它们进行分析。**严禁只修改参数**，必须在**结构、使用算子、或逻辑角度**上有所创新。

---
### **第一部分：因子表达式 (expression) - 核心要求：多样性**
*   **基础任务**: 将给定的自然语言描述，转化为一个单行的、可执行的因子表达式。
*   **铁律 1 (唯一允许的元素)**:
    *   **特征 (Features)**: `$open`, `$high`, `$low`, `$close`, `$volume`, `$openint`
    *   **算子 (Operators)**: {0}
*   **铁律 2 (绝对禁止的行为)**:
    *   **禁止发明特征**: 严禁使用任何不在允许列表中的特征，特别是 `$market_close` 等。
    *   **禁止使用未授权算子**: 必须严格使用上述算子列表。
*   **创造性指引 (Creative Guidance)**: 为了确保10个因子的多样性，你必须尝试从以下至少3-4个不同角度进行构思：
    *   **不同指标组合**: 尝试将思路与不同的基础指标（如均线、波动率、排名）结合。例如，不是直接计算量价关系，而是计算**均线处理后的量价关系**。
    *   **逻辑变形**: 原始思路是相乘？试试相减、相除或者构建条件逻辑（如使用 `Greater`, `Less`）。
    *   **引入非对称性**: 考虑上涨和下跌时量价关系的不同。例如，只在上涨时计算，或对上涨和下跌赋予不同权重。
    *   **标准化/归一化**: 尝试使用 `Std` (标准差) 或 `Rank` (排名) 对因子进行处理，以消除量纲影响或增强截面可比性。
    *   **不同价量特征**: 不要只用 `$close` 和 `$volume`。尝试引入 `$high`, `$low` 与成交量的关系，或者考虑日内振幅。

---
### **第二部分：详细逻辑解释 (explanation)**
*   **任务**: 详细解释**这个变体的独特之处**，说明它是如何从新角度诠释原始思路的。

---
### **第三部分：理论依据 (principle)**
*   **任务**: 阐述该因子变体背后独特的理论依据或市场假说。

---
### **第四部分：综合评分 (score)**
*   **任务**: 对该因子的逻辑性和**创新性**给出一个1到10的整数评分。
*   **评分标准**:
    *   **10分**: 逻辑清晰，理论扎实，且**创新性高**，提供了一个全新的视角。
    *   **5分**: 逻辑基本可行，但在结构上与原始思路相似，创新性一般。
    *   **1分**: 逻辑存在缺陷，或仅仅是微小的参数调整。

---
### **第五部分：最终输出格式 (Final Output Format)**
你的最终输出必须是一个包含`factors`列表的JSON对象，列表中必须恰好有10个因子。
"""

EXPS_USER_PROMPT = """
请严格遵循系统设定中的所有规则，特别是关于**创造性和多样性**的要求。

基于以下核心因子思路，请生成一个包含10个**结构不同**的因子变体的JSON输出。

**因子思路描述:**
> "{description}"

请生成最终的JSON输出:
{{

 "factors":[     
 {{
        "expression": "str"
        "explanation": "str",
        "principle":"str",
        "score": "int"
  }},
  {{
        "expression": "str"
        "explanation": "str",
        "principle":"str",
        "score": "int"
  }}
  ],
}}
"""

new_prompt = EXPS_SYSTEM_PROMPT.format(expression_str_md)


class DomInfo(BaseModel):
    expression: Optional[str] = Field(
        default="",  # 提供一个默认值 None
        description="因子表达式",
        define="",
    )
    explanation: Optional[str] = Field(
        default="",  # 提供一个默认值 None
        description="详细的逻辑解释",
        define="",
    )
    principle: Optional[str] = Field(
        default="",  # 提供一个默认值 None
        description="因子背后的理论依据",
        define="",
    )
    score: Optional[int] = Field(
        default="",  # 提供一个默认值 None
        description="<综合评分>",
        define="",
    )


class DomInfos(BaseModel):
    factors: List[DomInfo] = Field(
        ...,
        description="每个推理的因子详情",
        define="string",
    )


class FactorGenerator(object):

    def __init__(self, llm_provider, llm_model):
        self.llm_service = LLMServiceFactory.create_llm_service(
            llm_provider=llm_provider,
            llm_model=llm_model,
            system_message=new_prompt,
            other_parameters={'temperature': 0.0})

        self.message_generator = MessageGenerator(llm_service=self.llm_service,
                                                  debug_mode=DEBUG_MODE)

    async def generate(self,
                       description: Optional[str] = None) -> Dict[str, str]:
        content = await self.message_generator.generate_message_async(
            messages=[
                HumanMessage(content=EXPS_USER_PROMPT.format(
                    **{'description': description}))
            ],
            output_schema=DomInfos)
        return content.model_dump()


class ExpsFactorTool(BaseTool):
    name: str = "factor_exps_tool"
    description: str = "将一个自然语言的因子描述，转换为多个可计算的数学表达式。"
    parameters: dict = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string"
            }
        },
        "required": ["description"]
    }
    _factor_generator: Optional[FactorGenerator] = None

    def _initialize(self):
        if self._factor_generator is None:
            self._factor_generator = FactorGenerator(
                llm_model=os.environ['MODEL_NAME'],
                llm_provider=os.environ['MODEL_PROVIDER'])

    async def execute(self, description: str) -> ToolResult:
        try:
            self._initialize()
            expr = await self._factor_generator.generate(
                description=description)
            return ToolResult(output={"expression": expr})
        except Exception as e:
            return ToolResult(error=f"表达式生成失败: {e}")
