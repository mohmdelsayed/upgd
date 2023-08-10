from core.learner.learner import Learner
from core.optim.bgd import ExtendedBGD

class BGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = ExtendedBGD
        name = "bgd"
        super().__init__(name, network, optimizer, optim_kwargs)