import os, pdb
from abc import ABC, abstractmethod
from dataclasses import dataclass
from kdutils.model import Factor
from kdutils.until import create_memory_path


@dataclass(frozen=True)
class TrainData:
    short_prompt: str
    mid_prompt: str
    long_prompt: str
    reflection_prompt: str
    factors_details: str


class Trainer(ABC):

    def __init__(self, symbol, agent_class, portfolio=None):
        self.symbol = symbol
        self.portfolio = portfolio
        self.agent_class = agent_class
        self.portfolio = portfolio

    def initialize_agent(self, config_path, memory_path, date=None):
        checkpoint_path = None  
        # 步骤 1: 确定是否可以从检查点加载
        if isinstance(date, str) and date:  # 确保 date 是一个非空字符串
            path_to_check = create_memory_path(base_path=memory_path,
                                               date=date)
            if os.path.exists(path_to_check):
                checkpoint_path = path_to_check  # 如果路径存在，则设定它为要加载的路径

        # 步骤 2: 根据路径是否存在，执行加载或创建操作
        if checkpoint_path:
            # 如果确定了有效的检查点路径，则加载
            print(
                f"Loading agent from checkpoint: {checkpoint_path}")  # 建议增加日志
            self.agent = self.agent_class.load_checkpoint(
                path=checkpoint_path, config_path=config_path)
        else:
            # 否则 (date未提供、或路径不存在)，创建全新的 agent
            agent_config_path = os.path.join(config_path,
                                             self.agent_class.name)
            print(f"Creating new agent from config: {agent_config_path}"
                  )  # 建议增加日志
            self.agent = self.agent_class.from_config(path=agent_config_path)

        self.memory_path = memory_path

    @property
    def name(self):
        return self.agent.name

    @abstractmethod
    def prepare_data(self):
        """
        准备Agent所需的所有因子数据。
        这些数据在训练循环开始前一次性加载。
        """
        raise NotImplementedError(
            "Subclasses must implement _prepare_factor_data")

    @abstractmethod
    def create_group(self, date):
        """
        为给定的日期创建并返回一个 FactorsGroup 实例。
        """
        raise NotImplementedError(
            "Subclasses must implement _create_factors_group_for_date")

    ## 拆分步骤 用于异步处理
    def create_model(self, date, factors, desc, model_class):
        factors_list = model_class(date=date, desc=desc)
        for k, v in factors.items():
            factor_instance = Factor(name=k,
                                     value=float(v.loc[date].values[0]),
                                     desc=v.desc)
            factors_list.factors[k] = factor_instance
        return factors_list

    def handing_data(self, trade_date, factors_group):
        self.agent.handing_data(trade_date=trade_date,
                                symbol=self.symbol,
                                factors_group=factors_group)

    def query_records(self, trade_date):
        return self.agent.query_records(trade_date=trade_date,
                                        symbol=self.symbol)

    def create_data(self, date):
        factors_group = self.create_group(date)
        if not factors_group:
            print(f"Skipping date {date} due to missing factor data.")
            return
        self.handing_data(trade_date=date, factors_group=factors_group)
        long_prompt, mid_prompt, short_prompt, reflection_prompt = self.agent.query_records(
            trade_date=date, symbol=self.symbol)

        factors_details = factors_group.markdown(include_value=False)
        return TrainData(long_prompt=long_prompt,
                         mid_prompt=mid_prompt,
                         short_prompt=short_prompt,
                         reflection_prompt=reflection_prompt,
                         factors_details=factors_details)

    async def agenerate_suggestion(self, date, train_data, returns):
        response = await self.agent.agenerate_suggestion(
            date=date,
            symbol=self.symbol,
            short_prompt=train_data.short_prompt,
            mid_prompt=train_data.mid_prompt,
            long_prompt=train_data.long_prompt,
            reflection_prompt=train_data.reflection_prompt,
            factors_details=train_data.factors_details,
            returns=returns)
        response.name = self.agent.name
        return response

    def train(self, date, returns):
        factors_group = self.create_group(begin_date=date, end_date=date)
        if not factors_group:
            print(f"Skipping date {date} due to missing factor data.")
            return
        self.agent.handing_data(trade_date=date,
                                symbol=self.symbol,
                                factors_group=factors_group)

        long_prompt, mid_prompt, short_prompt, reflection_prompt = self.agent.query_records(
            trade_date=date, symbol=self.symbol)

        factors_details = factors_group.markdown(include_value=False)
        response = self.agent.generate_suggestion(
            date=date,
            symbol=self.symbol,
            short_prompt=short_prompt,
            mid_prompt=mid_prompt,
            long_prompt=long_prompt,
            reflection_prompt=reflection_prompt,
            factors_details=factors_details,
            returns=returns)
        return response

    async def atrain(self, date, returns):
        factors_group = self.create_group(date)
        if not factors_group:
            print(f"Skipping date {date} due to missing factor data.")
            return

        self.handing_data(trade_date=date,
                          symbol=self.symbol,
                          factors_group=factors_group)

        long_prompt, mid_prompt, short_prompt, reflection_prompt = self.agent.query_records(
            trade_date=date, symbol=self.symbol)

        factors_details = factors_group.markdown(include_value=False)

        response = await self.agent.agenerate_suggestion(
            date=date,
            symbol=self.symbol,
            short_prompt=short_prompt,
            mid_prompt=mid_prompt,
            long_prompt=long_prompt,
            reflection_prompt=reflection_prompt,
            factors_details=factors_details,
            returns=returns)
        return response

    def update_memory(self, date, returns, response, is_save=True):
        actions = 1 if returns > 0 else -1
        feedback = {"feedback": actions, "date": date}
        self.agent.update_memory(trade_date=date,
                                 symbol=self.symbol,
                                 response=response,
                                 feedback=feedback)

        if is_save:
            self.agent.save_checkpoint(path=create_memory_path(
                base_path=self.memory_path, date=date),
                                       force=True)
