from core.learner.learner import Learner
from core.optim.adaptive_pgd import AdaptivePGD

class AdaptivePGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaptivePGD
        name = "adaptive_pgd"
        super().__init__(name, network, optimizer, optim_kwargs)