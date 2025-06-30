import os
from abc import ABC, abstractmethod
from dichaos.agents.indexor.porfolio import Portfolio
from motvi.kdutils.model import Factor


class Predictor(ABC):

    def __init__(self, symbol, agent_class):
        self.symbol = symbol
        self.portfolio = Portfolio(symbol=symbol, lookback_window_size=0)
        self.agent_class = agent_class

    def initialize_agent(self, base_path, symbol, date):
        self.agent = self.agent_class.load_checkpoint(path=os.path.join(
            base_path, self.agent_class.name, f'{symbol}_{date}'))

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

    def create_model(self, date, factors, desc, model_class):
        factors_list = model_class(date=date, desc=desc)
        for k, v in factors.items():
            factor_instance = Factor(name=k,
                                     value=float(v.loc[date].values[0]),
                                     desc=v.desc)
            factors_list.factors[k] = factor_instance
        return factors_list

    def prdict(self, date, future_data):
        factors_group = self.create_group(date)
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
            returns=future_data['returns'].values[0])
