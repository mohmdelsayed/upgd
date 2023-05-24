from core.learner.learner import Learner
from core.optim.feature.search.first_order import FirstOrderSearchLocalAnticorrelated, FirstOrderSearchLocalUncorrelated, FirstOrderSearchGlobalAnticorrelated, FirstOrderSearchGlobalUncorrelated
from core.optim.feature.search.second_order import SecondOrderSearchLocalAnticorrelated, SecondOrderSearchLocalUncorrelated, SecondOrderSearchGlobalAnticorrelated, SecondOrderSearchGlobalUncorrelated

class FeatureFirstOrderSearchLocalAnticorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchLocalAnticorrelated
        name = "feature_search_fo_anti_corr_local"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureFirstOrderSearchLocalUncorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchLocalUncorrelated
        name = "feature_search_fo_uncorr_local"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureFirstOrderSearchGlobalAnticorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchGlobalAnticorrelated
        name = "feature_search_fo_anti_corr_global"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureFirstOrderSearchGlobalUncorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchGlobalUncorrelated
        name = "feature_search_fo_uncorr_global"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureSecondOrderSearchLocalAnticorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderSearchLocalAnticorrelated
        name = "feature_search_so_anti_corr_local"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureSecondOrderSearchLocalUncorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderSearchLocalUncorrelated
        name = "feature_search_so_uncorr_local"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureSecondOrderSearchGlobalAnticorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderSearchGlobalAnticorrelated
        name = "feature_search_so_anti_corr_global"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureSecondOrderSearchGlobalUncorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderSearchGlobalUncorrelated
        name = "feature_search_so_uncorr_global"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)