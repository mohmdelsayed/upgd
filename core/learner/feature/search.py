from core.learner.learner import Learner
from core.optim.feature.search.first_order import FirstOrderSearchAntiCorrNormalized, FirstOrderSearchNormalNormalized, FirstOrderSearchAntiCorrMax, FirstOrderSearchNormalMax
from core.optim.feature.search.second_order import SecondOrderSearchAntiCorrNormalized, SecondOrderSearchNormalNormalized, SecondOrderSearchAntiCorrMax, SecondOrderSearchNormalMax

class FeatureSearchLearnerAntiCorrFONormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchAntiCorrNormalized
        name = "feature_search_fo_anti_corr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureSearchLearnerNormalFONormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchNormalNormalized
        name = "feature_search_fo_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureSearchLearnerAntiCorrFOMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchAntiCorrMax
        name = "feature_search_fo_anti_corr_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureSearchLearnerNormalFOMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchNormalMax
        name = "feature_search_fo_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureSearchLearnerAntiCorrSONormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderSearchAntiCorrNormalized
        name = "feature_search_so_anti_corr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureSearchLearnerNormalSONormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderSearchNormalNormalized
        name = "feature_search_so_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureSearchLearnerAntiCorrSOMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderSearchAntiCorrMax
        name = "feature_search_so_anti_corr_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureSearchLearnerNormalSOMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderSearchNormalMax
        name = "feature_search_so_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)