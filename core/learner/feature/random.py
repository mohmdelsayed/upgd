from core.learner.learner import Learner
from core.optim.feature.search.random import RandomSearchNormal, RandomSearchAntiCorr

class FeatureRandomSearchLearnerNormal(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = RandomSearchNormal
        name = "feature_random_search_normal"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureRandomSearchLearnerAntiCorr(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = RandomSearchAntiCorr
        name = "feature_random_search_anti_corr"
        super().__init__(name, network, optimizer, optim_kwargs)