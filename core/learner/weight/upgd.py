from core.learner.learner import Learner
from core.optim.weight.upgd.first_order import FirstOrderLocalUPGD, FirstOrderNonprotectingLocalUPGD, FirstOrderGlobalUPGD, FirstOrderNonprotectingGlobalUPGD
from core.optim.weight.upgd.second_order import SecondOrderLocalUPGD, SecondOrderNonprotectingLocalUPGD, SecondOrderGlobalUPGD, SecondOrderNonprotectingGlobalUPGD

class FirstOrderLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderLocalUPGD
        name = "upgd_v2_fo_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderLocalUPGD
        name = "upgd_v2_so_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FirstOrderNonprotectingLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderNonprotectingLocalUPGD
        name = "upgd_v1_fo_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderNonprotectingLocalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderNonprotectingLocalUPGD
        name = "upgd_v1_so_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FirstOrderGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderGlobalUPGD
        name = "upgd_v2_fo_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderGlobalUPGD
        name = "upgd_v2_so_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FirstOrderNonprotectingGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderNonprotectingGlobalUPGD
        name = "upgd_v1_fo_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class SecondOrderNonprotectingGlobalUPGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderNonprotectingGlobalUPGD
        name = "upgd_v1_so_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)