from core.learner.learner import Learner
from core.optim.weight.search.random import RandomSearchNormal, RandomSearchAntiCorr

class RandomSearchLearnerNormal(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = RandomSearchNormal
        name = "random_search_normal"
        super().__init__(name, network, optimizer, optim_kwargs)

class RandomSearchLearnerAntiCorr(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = RandomSearchAntiCorr
        name = "random_search_anti_corr"
        super().__init__(name, network, optimizer, optim_kwargs)