from core.learner.learner import Learner
from core.optim.adaptive_anti_pgd import AdaptiveAntiPGD

class AdaptiveAntiPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaptiveAntiPGD
        name = "adaptive_anti_pgd"
        super().__init__(name, network, optimizer, optim_kwargs)