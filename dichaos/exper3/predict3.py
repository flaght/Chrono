import os, pdb
from dotenv import load_dotenv

load_dotenv()

from alphacopilot.calendars.api import advanceDateByCalendar
from motvi.factors.extractor import fetch_main_returns
from motvi.agent.decision.predict import Predictor as DecisionPredict
from motvi.agent.indicator.predict import Predictor as IndicatorPredict
from motvi.agent.posflow.predict import Predictor as PosflowPredict
from motvi.kdutils.notice import feishu


def create_data(begin_date, end_date, codes):
    returns_data = fetch_main_returns(begin_date=begin_date,
                                      end_date=end_date,
                                      codes=codes,
                                      columns=None)
    return returns_data


end_date = '2025-07-10'
begin_date = advanceDateByCalendar('china.sse', end_date,
                                   '-{0}b'.format(10)).strftime('%Y-%m-%d')
symbol = 'IM'

base_path = os.path.join(os.environ['BASE_PATH'], 'memory')

decision_predict = DecisionPredict(symbol=symbol,
                                   base_path=base_path,
                                   date='2025-01-16')

decision_predict.add_sub_agent(
    IndicatorPredict(symbol=symbol,
                     base_path=os.path.join(os.environ['BASE_PATH'], 'memory'),
                     date='2025-01-09'))

decision_predict.add_sub_agent(
    PosflowPredict(symbol=symbol,
                   base_path=os.path.join(os.environ['BASE_PATH'], 'memory'),
                   date='2025-01-08'))

decision_predict.prepare_data(begin_date=begin_date, end_date=end_date)

response = decision_predict.predict(date=end_date)

content = "{0}: 日期:{1}, 方向:{2}\n\n推理原因:{3}\n\n决策的核心:{4}".format(
    symbol,
    advanceDateByCalendar('china.sse', end_date, '1b').strftime('%Y-%m-%d'),
    response.signal, response.reasoning, response.analysis_details)
print(content)
feishu(content)
