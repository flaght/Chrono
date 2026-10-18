from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional, List


class NewsEvent(BaseModel):
    event_id: str
    # 首选 summary；兼容部分模型返回 title + description 的情况。
    summary: Optional[str] = None
    title: Optional[str] = None
    category: Optional[str] = None
    status: Literal[
        "confirmed",
        "planned",
        "forecast",
        "opinion",
        "unverified",
        "ambiguous",
    ]
    importance: int = Field(ge=1, le=5)

    @model_validator(mode="after")
    def normalize_summary(self):
        if not self.summary:
            self.summary = self.description or self.title
        if not self.summary:
            raise ValueError("事件必须提供 summary 或 description/title")
        return self


class NewsEventResult(BaseModel):
    events: List[NewsEvent]


class TextualFeature(BaseModel):
    feature_id: str = Field(
        pattern=r"^TF-\d{8}-\d{3}$",
        description="格式为TF-YYYYMMDD-三位序号",
    )

    feature_type: Literal["domestic_policy", "domestic_macro",
                          "global_liquidity", "external_shock",
                          "systemic_risk", "other"]

    affected_scope: Literal["broad_market", "style", "major_industry",
                            "local_or_narrow"]

    status: Literal["confirmed", "planned", "forecast", "opinion",
                    "unverified", "ambiguous"]
    digest: str = Field(
        ...,
        min_length=3,
        max_length=40,
        description=
        "极简、高度压缩的原子事件短语侧重在核心描述，建议10~25字，专用于特征池展示与Prompt紧凑注入",
    )
    summary: str = Field(
        min_length=1,
        max_length=200,
        description="客观、完整且可以独立阅读的事件摘要 建议不超过100个汉字，最多150个字符",
    )

    importance: int = Field(ge=1, le=5)


class TextualFeatureResult(BaseModel):
    textuals: List[TextualFeature]


# ----------------------------------------------------------------------
# 1. 单条因果推演逻辑链条 (Predict Logic Item)
# ----------------------------------------------------------------------
class PredictLogicItem(BaseModel):
    feature: str = Field(
        ...,
        description=
        "引用的特征变量名：若为 PREDICT/REGIME 填写具体因子名（如 main_flow_ratio, total_market_turnover）；若为 TEXTUAL 必须从 6 大稳定类别中选择一个：['domestic_macro', 'domestic_policy', 'global_liquidity', 'external_shock', 'industry_trend', 'market_sentiment']",
    )
    feature_layer: Literal["PREDICT", "REGIME", "TEXTUAL"] = Field(
        ...,
        description="特征所属物理层级：PREDICT(预测特征层), REGIME(市场特征层), TEXTUAL(文本特征层)",
    )
    observation: str = Field(
        ...,
        description="在时序回溯窗口中观察到的客观事实、演化形态或数值拐点（如：主力资金连续2日净流入且加速度由负转正）",
    )
    inference: str = Field(
        ...,
        description="基于该事实推导出的微观多空逻辑结论（如：资金买盘主动性增强，短期惯性上冲概率高）",
    )


# ----------------------------------------------------------------------
# 2. 采信的历史案例记忆独立声明 (Used Memory Item)
# ----------------------------------------------------------------------
class UsedMemoryItem(BaseModel):
    memory_id: str = Field(
        ...,
        description="实际采信的历史经验 ID（如 MEM-20260715-001）",
    )
    implied_direction: Literal["UP", "DOWN", "FLAT"] = Field(
        ...,
        description="该条经验所独立支持的操作方向",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="对该条经验独立分配的采信置信度 (0.0 ~ 1.0)",
    )
    usage: Literal["primary", "auxiliary"] = Field(
        ...,
        description="记忆的主辅关系定位：primary(核心主记忆，必须且只能有1条), auxiliary(辅助参考记忆)",
    )


# ----------------------------------------------------------------------
# 3. 终极交易决策生成容器 (Trader Prediction Result)
# ----------------------------------------------------------------------
class TraderPredictionResult(BaseModel):
    """
    Trader 预测 Agent 的前瞻多空推演与交易决策生成统一响应结构
    （双态兼容：训练事前零样本盲测态输出 used_memories 为空列表 []；实盘/验证态输出采信的 used_memories）
    """
    dominant_regime: str = Field(
        ...,
        description="对当前市场所处环境 Regime 底色的一句话定性（如：高波宽幅震荡筑底期、单边主升趋势等）",
    )
    predict_direction: Literal["UP", "DOWN", "FLAT"] = Field(
        ...,
        description="前瞻持有周期内的最终预测方向：UP(看多做多), DOWN(看空做空), FLAT(观望/信号冲突)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="整体交易决策的综合置信度 (0.0 ~ 1.0)",
    )
    used_features: List[str] = Field(
        ...,
        description="支撑本次决策的核心预测特征名或 TF-ID 列表（作为底层特征归因与信用分配的锚点）",
    )
    predict_logic: List[PredictLogicItem] = Field(
        ...,
        min_items=1,
        description="支撑本次决策的标准因果逻辑链条列表 (Feature -> Observation -> Inference)",
    )
    used_memories: List[UsedMemoryItem] = Field(
        default_factory=list,
        description="实际采信并对齐的历史案例记忆声明列表（盲测态固定为空列表 []）",
    )
    invalid_conditions: List[str] = Field(
        ...,
        min_items=1,
        description="导致本次预测逻辑推演彻底失效的盘面触发现象列表（红线止损/反转条件）",
    )


# ----------------------------------------------------------------------
# 模块 1: 前置预测问题归因 (Prediction Attribution)
# ----------------------------------------------------------------------
class PredictionAttribution(BaseModel):
    prediction_outcome: Literal["WIN", "LOSS", "NEUTRAL", "FLAT_CORRECT",
                                "FLAT_MISSED"] = Field(
                                    ...,
                                    description="预测胜负结果 (只读继承自 settlement)")
    root_cause_analysis: str = Field(
        ...,
        description="核心问题归因：Trader 在事前判断对/错的根本原因",
    )
    logic_flaws: List[str] = Field(
        default_factory=list,
        description="事前逻辑漏洞列表（如数据误读、动能过度外推、Regime与信号错配）",
    )
    missed_pre_t_evidences: List[str] = Field(
        default_factory=list,
        description="事前（时点 T 之前）已经存在但被 Trader 忽略的关键风险/反转线索",
    )


# ----------------------------------------------------------------------
# 模块 2: 结构化原子证据单元与决策规则 (Structured Experience Summary)
# ----------------------------------------------------------------------
class EvidenceUnit(BaseModel):
    evidence_id: str = Field(
        ...,
        description="证据单元唯一ID (如 EV1, EV2, EV3)",
    )
    feature_refs: List[str] = Field(
        ...,
        min_items=1,
        description="该证据所依赖的一个或多个稳定特征变量名。"
        "若 feature_layer 为 PREDICT/REGIME，填写具体因子名（如 ['basis_pct'] 或 ['main_flow_ratio']）；"
        "若 feature_layer 为 TEXTUAL，必须且只能从 6 大稳定文本类别中选择："
        "['domestic_macro', 'domestic_policy', 'global_liquidity', 'external_shock', 'industry_trend', 'market_sentiment']",
    )
    feature_layer: Literal["PREDICT", "REGIME", "TEXTUAL"] = Field(
        ...,
        description="特征所属物理层级：PREDICT(预测特征层), REGIME(市场特征层), TEXTUAL(文本特征层)",
    )
    interpretation: str = Field(
        ...,
        description="基于上述特征组合所观察到的客观事实与微观博弈解释",
    )
    evidence_role: Literal[
        "DIRECTIONAL", "CONFIRMATION", "BACKGROUND", "AUXILIARY"] = Field(
            ...,
            description=
            "证据角色定位：DIRECTIONAL(核心方向证据), CONFIRMATION(确认证据), BACKGROUND(环境背景), AUXILIARY(次要辅助)",
        )


class DecisionRule(BaseModel):
    required_evidence: List[str] = Field(
        ...,
        min_items=1,
        description="本经验成立的核心必要证据ID列表（如 ['EV2']），若其中特征被淘汰则经验直接熔断",
    )
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="本经验的辅助支撑证据ID列表（如 ['EV1', 'EV3']），若其中特征被淘汰可降级生成投影经验",
    )
    conclusion: str = Field(
        ...,
        description="基于上述证据组合推导出的最终操作决策结论（如：下行风险显著增加，应降低多头敞口转为防守/偏空）",
    )


class InvalidationRule(BaseModel):
    feature_refs: List[str] = Field(
        ...,
        description="触发失效监控的关键特征列表",
    )
    condition: str = Field(
        ...,
        description="具体的盘面失效触发条件描述",
    )
    conclusion: str = Field(
        ...,
        description="失效后的应对结论（如：下行经验失效，恢复中性观望）",
    )


class StructuredExperienceSummary(BaseModel):
    implied_direction: Literal["UP", "DOWN", "FLAT"] = Field(
        ...,
        description="本次复盘总结后修正的操作指向：UP(多), DOWN(空), FLAT(观望/防守)",
    )
    evidence_units: List[EvidenceUnit] = Field(
        ...,
        min_items=1,
        description="结构化原子证据单元列表 (事实真相来源，支持多特征与单特征依赖)",
    )
    decision_rule: DecisionRule = Field(
        ...,
        description="证据组合与决策合成规则 (明确声明必要证据与辅助证据)",
    )
    invalidation_rules: List[InvalidationRule] = Field(
        default_factory=list,
        description="失效证伪规则列表",
    )
    validation_status: Literal["UNVALIDATED_SINGLE_CASE"] = Field(
        "UNVALIDATED_SINGLE_CASE",
        description="初始固定为 UNVALIDATED_SINGLE_CASE",
    )


# ----------------------------------------------------------------------
# 终极容器：CaseMemoryReviewResult (v4.0 结构化原子证据版)
# ----------------------------------------------------------------------
class CaseMemoryReviewResult(BaseModel):
    """
    Train Engine 事后反思 Reviewer Agent 的标准化输出模型 (v4.0 结构化证据版)
    """
    case_id: str = Field(..., description="案例唯一标识")
    data_quality_flags: List[str] = Field(
        default_factory=list,
        description="数据异常或冲突标记（若无则为空列表）",
    )
    prediction_attribution: PredictionAttribution = Field(
        ...,
        description="1. 前置预测结果的问题归因",
    )
    experience_summary: StructuredExperienceSummary = Field(
        ...,
        description="2. 结构化原子证据单元与决策规则提炼",
    )
