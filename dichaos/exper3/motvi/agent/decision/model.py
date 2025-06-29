from motvi.kdutils.model import *
import pdb, json


class AgentsResult(BaseModel):
    date: str  ## 日期
    desc: str  ## agent描述
    reasoning: str  ## 决策逻辑
    confidence: int  ## 置信度
    singal: str  ## 信号
    analysis_details: str  ## 关键特征分析

    def format(self, include_head=False, to_text=False):
        dis = {}
        if include_head:
            dis['trade_date'] = self.date
        dis['desc'] = self.desc
        dis['reasoning'] = self.reasoning
        dis['confidence'] = self.confidence
        dis['signal'] = self.singal
        dis['analysis_details'] = self.analysis_details
        return json.dumps(dis) if to_text else dis


class AgentsGroup(BaseModel):
    date: str
    symbol: str
    index: Optional[str] = Field(default=None, nullable=True)
    agents_list: List[AgentsResult] = Field(default_factory=list)

    def format(self, types: str = "short"):
        desc = {"short": "短期记忆", "mid": "中期记忆", "long": "长期记忆"}
        sidx = {"short": "S", "mid": "M", "long": "L"}
        texts = "{0}索引ID: {1}{2}\n".format(desc[types], sidx[types],
                                           self.index)

        res = []
        for result in self.agents_list:
            res.append(result.format())

        texts += json.dumps(res)
        return texts
