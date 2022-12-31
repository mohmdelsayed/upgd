from core.learner.learner import Learner
from core.optim.anti_pgd import ExtendedAntiPGD

class AntiPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = ExtendedAntiPGD
        name = "anti_pgd"
        super().__init__(name, network, optimizer, optim_kwargs)