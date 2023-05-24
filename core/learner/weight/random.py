from core.learner.learner import Learner
from core.optim.weight.search.random import RandomSearchUncorrelated, RandomSearchAnticorrelated

class RandomSearchUncorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = RandomSearchUncorrelated
        name = "random_search_normal"
        super().__init__(name, network, optimizer, optim_kwargs)

class RandomSearchAnticorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = RandomSearchAnticorrelated
        name = "random_search_anti_corr"
        super().__init__(name, network, optimizer, optim_kwargs)