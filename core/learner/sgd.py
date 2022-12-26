from core.learner.learner import Learner
from core.optim.gt.first_order import ExtendedSGD


class SGDLearner(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = ExtendedSGD
        name = "sgd"
        super().__init__(name, network, optimizer, optim_kwargs)
