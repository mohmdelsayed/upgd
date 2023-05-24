from core.learner.learner import Learner
from core.optim.weight.search.first_order import FirstOrderSearchLocalAnticorrelated, FirstOrderSearchLocalUncorrelated, FirstOrderSearchGlobalUncorrelated, FirstOrderSearchGlobalAnticorrelated
from core.optim.weight.search.second_order import SecondOrderSearchLocalAnticorrelated, SecondOrderSearchLocalUncorrelated, SecondOrderSearchGlobalUncorrelated, SecondOrderSearchGlobalAnticorrelated


class FirstOrderSearchLocalUncorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchLocalUncorrelated
        name = "search_fo_uncorr_local"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderSearchLocalUncorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderSearchLocalUncorrelated
        name = "search_so_uncorr_local"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FirstOrderSearchLocalAnticorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchLocalAnticorrelated
        name = "search_fo_anti_corr_local"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderSearchLocalAnticorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderSearchLocalAnticorrelated
        name = "search_so_anti_corr_local"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FirstOrderSearchGlobalUncorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchGlobalUncorrelated
        name = "search_fo_uncorr_global"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderSearchGlobalUncorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderSearchGlobalUncorrelated
        name = "search_so_uncorr_global"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FirstOrderSearchGlobalAnticorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderSearchGlobalAnticorrelated
        name = "search_fo_anti_corr_global"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderSearchGlobalAnticorrelatedLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderSearchGlobalAnticorrelated
        name = "search_so_anti_corr_global"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)