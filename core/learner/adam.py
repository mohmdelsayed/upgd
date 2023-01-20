from core.learner.learner import Learner
from core.optim.adam import ExtendedAdam

class AdamLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = ExtendedAdam
        name = "adam"
        super().__init__(name, network, optimizer, optim_kwargs)