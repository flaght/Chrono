#### 决策agent

import os, pdb
from dotenv import load_dotenv

load_dotenv()
from motvi.agent.indicator.predict import Predictor as IndicatorPredict
from motvi.agent.posflow.predict import Predictor as PosflowPredict
from motvi.factors.extractor import fetch_main_returns
from motvi.agent.decision.agent import Agent
from motvi.agent.decision.model import *

def create_data(begin_date, end_date, codes):
    returns_data = fetch_main_returns(begin_date=begin_date,
                                      end_date=end_date,
                                      codes=codes,
                                      columns=None)
    return returns_data


begin_date = '2025-01-02'
end_date = '2025-02-01'
symbol = 'IM'

returns_data = create_data(begin_date, end_date, codes=[symbol])

predict_pool = [
    IndicatorPredict(symbol=symbol,
                     base_path=os.path.join(os.environ['BASE_PATH'], 'memory'),
                     date='2025-01-09'),
    PosflowPredict(symbol=symbol,
                   base_path=os.path.join(os.environ['BASE_PATH'], 'memory'),
                   date='2025-01-08')
]

for predictor in predict_pool:
    predictor.prepare_data(begin_date=begin_date, end_date=end_date)

trade_date = '2025-01-14'
decision_agent = Agent.from_config(
    path=os.path.join("motvi", "agent", Agent.name))


future_data = returns_data.loc[trade_date]

agents_group = AgentsGroup(date=trade_date, symbol=symbol)
for predictor in predict_pool:
    response = predictor.predict(date=trade_date)
    result = AgentsResult(date=trade_date,
                          symbol=symbol,
                          name=predictor.agent.name,
                          desc=predictor.agent.desc(),
                          reasoning=response.reasoning,
                          confidence=response.confidence,
                          signal=response.signal,
                          analysis_details=response.analysis_details)
    agents_group.agents_list.append(result)

decision_agent.handing_data(trade_date=trade_date,
                            symbol=symbol,
                            agent_group=agents_group)
long_prompt, mid_prompt, short_prompt, reflection_prompt = decision_agent.query_records(
    trade_date=trade_date, symbol=symbol)

response = decision_agent.generate_suggestion(
    date=trade_date, symbol=symbol,
    short_prompt=short_prompt,
    mid_prompt=mid_prompt,
    long_prompt=long_prompt,
    reflection_prompt=reflection_prompt,
    returns=future_data['returns'].values[0]
)
decision_agent.save_checkpoint(path=os.path.join(os.environ['BASE_PATH'],
                                                'memory', decision_agent.name,
                                                f'{symbol}_{trade_date}'),
                              force=True)
pdb.set_trace()
print('-->')