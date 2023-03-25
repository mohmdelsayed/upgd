from core.learner.learner import Learner
from core.optim.shrink_and_perturb import ExtendedShrinkandPerturb

class ShrinkandPerturbLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = ExtendedShrinkandPerturb
        name = "shrink_and_perturb"
        super().__init__(name, network, optimizer, optim_kwargs)