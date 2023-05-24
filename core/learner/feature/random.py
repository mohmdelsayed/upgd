from core.learner.learner import Learner
from core.optim.feature.search.random import RandomSearchUncorrelated, RandomSearchAnticorrelated

class FeatureRandomSearchUncorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = RandomSearchUncorrelated
        name = "feature_random_search_normal"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureRandomSearchAnticorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = RandomSearchAnticorrelated
        name = "feature_random_search_anti_corr"
        super().__init__(name, network, optimizer, optim_kwargs)