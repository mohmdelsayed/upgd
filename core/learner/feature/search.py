from core.learner.learner import Learner
from core.optim.feature.search.first_order import FirstOrderSearchAntiCorrNormalized, FirstOrderSearchNormalNormalized, FirstOrderSearchAntiCorrMax, FirstOrderSearchNormalMax

class FeatureSearchLearnerAntiCorrFONormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchAntiCorrNormalized
        name = "feature_search_fo_anticorr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureSearchLearnerNormalFONormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchNormalNormalized
        name = "feature_search_fo_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureSearchLearnerAntiCorrFOMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchAntiCorrMax
        name = "feature_search_fo_anticorr_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureSearchLearnerNormalFOMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchNormalMax
        name = "feature_search_fo_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs)