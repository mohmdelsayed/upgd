from core.learner.learner import Learner
from core.optim.search.first_order import FirstOrderSearchAntiCorrNormalized, FirstOrderSearchNormalNormalized, FirstOrderSearchNormalMax, FirstOrderSearchAntiCorrMax
from core.optim.search.second_order import SecondOrderSearchAntiCorrNormalized, SecondOrderSearchNormalNormalized, SecondOrderSearchNormalMax, SecondOrderSearchAntiCorrMax


class SearchLearnerNormalFONormalized(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderSearchNormalNormalized
        name = "search_fo_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class SearchLearnerNormalSONormalized(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = SecondOrderSearchNormalNormalized
        name = "search_so_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class SearchLearnerAntiCorrFONormalized(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderSearchAntiCorrNormalized
        name = "search_fo_anticorr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class SearchLearnerAntiCorrSONormalized(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = SecondOrderSearchAntiCorrNormalized
        name = "search_so_anticorr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class SearchLearnerNormalFOMax(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderSearchNormalMax
        name = "search_fo_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class SearchLearnerNormalSOMax(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = SecondOrderSearchNormalMax
        name = "search_so_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class SearchLearnerAntiCorrFOMax(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderSearchAntiCorrMax
        name = "search_fo_anticorr_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class SearchLearnerAntiCorrSOMax(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = SecondOrderSearchAntiCorrMax
        name = "search_so_anticorr_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)