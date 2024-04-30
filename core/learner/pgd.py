from core.learner.learner import Learner
from core.optim.pgd import PGD

class PGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = PGD
        name = "pgd"
        super().__init__(name, network, optimizer, optim_kwargs)