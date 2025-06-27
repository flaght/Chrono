from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from motvi.kdutils.model import *
import pdb


class FactorsList(BaseModel):
    date: str  ## 日期
    desc: str  ## 描述
    ## 动态存储因子 键是因子ID
    factors: Dict[str, Factor] = Field(default_factory=dict)

    def text(self, include_head=False):
        texts = ""
        if include_head:
            texts += f"{self.desc} \n"
        texts += "{0} 因子数值:\n".format(self.date)
        for factor_id, factor_data in self.factors.items():
            # 格式化输出，保留4位小数
            texts += f"{factor_data.name}: {factor_data.value:.4f}\n"
        return texts

    def desc_to_string(self):
        # 使用列表推导式高效地生成每一行
        desc_lines = [
            f"{factor.name}:{factor.desc}" for factor in self.factors.values()
        ]

        # 使用换行符将所有行连接成一个字符串
        return "\n".join(desc_lines)

    def markdown(self, include_value):
        if include_value:
            header = "| 因子名字 | 因子值 | 因子描述 |"
            separator = "| :--- | :--- | :--- |"
        else:
            header = "| 因子名字 | 因子描述 |"
            separator = "| :--- | :--- |"
        table_rows = [header, separator]

        # 遍历所有因子，为每个因子创建一行
        for factor in self.factors.values():
            if include_value:
                # 包含因子值，并格式化为4位小数
                row = f"| {factor.name} | {factor.value:.4f} | {factor.desc} |"
            else:
                # 不包含因子值
                row = f"| {factor.name} | {factor.desc} |"
            table_rows.append(row)

        # 使用换行符将所有行连接成一个完整的字符串
        return "\n".join(table_rows)


class FactorsGroup(BaseModel):
    date: str
    symbol: str
    index: Optional[str] = Field(default=None, nullable=True)
    factors_list: List[FactorsList] = Field(default_factory=list)

    def markdown(self, include_value):
        last_desc = ""
        texts = ""
        for factor_list in self.factors_list:
            current_desc = factor_list.markdown(include_value)
            if current_desc != last_desc:
                texts += current_desc
                texts += "\n"
        return texts

    def format(self, types: str = "short"):
        desc = {"short": "短期记忆", "mid": "中期记忆", "long": "长期记忆"}
        sidx = {"short": "S", "mid": "M", "long": "L"}
        texts = "{0}索引ID: {1}{2}\n".format(desc[types], sidx[types],
                                           self.index)
        for factor_list in self.factors_list:
            texts += factor_list.text(include_head=True)
        return texts