from core.learner.learner import Learner
from core.optim.ewc import EWC

class EWCLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = EWC
        name = "ewc"
        super().__init__(name, network, optimizer, optim_kwargs)
