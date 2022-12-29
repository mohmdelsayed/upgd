from core.learner.learner import Learner
from core.optim.search.first_order import FirstOrderSearchAntiCorr, FirstOrderSearchNormal
from core.optim.search.second_order import SecondOrderSearchAntiCorr, SecondOrderSearchNormal


class SearchLearnerNormalFO(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderSearchNormal
        name = "search_fo_normal"
        super().__init__(name, network, optimizer, optim_kwargs)

class SearchLearnerNormalSO(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = SecondOrderSearchNormal
        name = "search_so_normal"
        super().__init__(name, network, optimizer, optim_kwargs)

class SearchLearnerAntiCorrFO(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderSearchAntiCorr
        name = "search_fo_anticorr"
        super().__init__(name, network, optimizer, optim_kwargs)

class SearchLearnerAntiCorrSO(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = SecondOrderSearchAntiCorr
        name = "search_so_anticorr"
        super().__init__(name, network, optimizer, optim_kwargs)
