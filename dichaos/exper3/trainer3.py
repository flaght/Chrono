import os, pdb
from dotenv import load_dotenv

load_dotenv()

from motvi.factors.extractor import fetch_main_returns
from motvi.agent.decision.trainer import Trainer
from motvi.agent.indicator.predict import Predictor as IndicatorPredict
from motvi.agent.posflow.predict import Predictor as PosflowPredict


def create_data(begin_date, end_date, codes):
    returns_data = fetch_main_returns(begin_date=begin_date,
                                      end_date=end_date,
                                      codes=codes,
                                      columns=None)
    return returns_data


begin_date = '2025-01-02'
end_date = '2025-01-20'
symbol = 'IM'
base_path = os.path.join(os.environ['BASE_PATH'], 'memory')

futures_data = create_data(begin_date, end_date, [symbol])
pdb.set_trace()
trainer = Trainer(symbol=symbol, base_path=os.path.join("motvi", "agent"))

### 添加子策略
trainer.add_sub_agent(
    IndicatorPredict(symbol=symbol,
                     base_path=os.path.join(os.environ['BASE_PATH'], 'memory'),
                     date='2025-01-09'))

trainer.add_sub_agent(
    PosflowPredict(symbol=symbol,
                   base_path=os.path.join(os.environ['BASE_PATH'], 'memory'),
                   date='2025-01-08'))

trainer.prepare_data(begin_date=begin_date, end_date=end_date)

dates = ['2025-01-13', '2025-01-14', '2025-01-15', '2025-01-16']
for d1 in dates:
    response = trainer.train(date=d1,
                  future_data=futures_data.loc[d1])
