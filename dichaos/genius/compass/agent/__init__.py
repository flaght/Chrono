from agent.clouto.predict import Predictor as CloutoPredict
from agent.indicator.predict import Predictor as IndicatorPredict
from agent.moneyflow.predict import Predictor as MoneyFlowPredict
from agent.posflow.predict import Predictor as PosFlowPredict

from agent.clouto.train import Trainer as CloutoTrain
from agent.indicator.train import Trainer as IndicatorTrain
from agent.moneyflow.train import Trainer as MoneyFlowTrain
from agent.posflow.train import Trainer as PosFlowTrain

__all__ = [
    'CloutoPredict', 'IndicatorPredict', 'MoneyFlowPredict', 'PosFlowPredict',
    'CloutoTrain', 'IndicatorTrain', 'MoneyFlowTrain', 'PosFlowTrain'
]
