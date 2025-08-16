from agent.indicator.predict import Predictor as IndicatorPredict
from agent.moneyflow.predict import Predictor as MoneyFlowPredict
from agent.clouto.predict import Predictor as CloutoPredict
from agent.hotmoney.predict import Predictor as HotMoneyPredict
from agent.chip.predict import Predictor as ChipPredict

from agent.indicator.train import Trainer as IndicatorTrain
from agent.moneyflow.train import Trainer as MoneyFlowTrain
from agent.clouto.train import Trainer as CloutoTrain
from agent.hotmoney.train import Trainer as HotMoneyTrain
from agent.chip.train import Trainer as ChipTrain

__all__ = [
    'IndicatorPredict', 'MoneyFlowPredict', 'CloutoPredict', 'HotMoneyPredict',
    'ChipPredict', 'IndicatorTrain', 'MoneyFlowTrain', 'CloutoTrain',
    'HotMoneyTrain', 'ChipTrain'
]
