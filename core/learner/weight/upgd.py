from core.learner.learner import Learner
from core.optim.weight.gt.first_order import FirstOrderUPGDv2AntiCorrNormalized, FirstOrderUPGDv1AntiCorrNormalized, FirstOrderUPGDv2AntiCorrMax, FirstOrderUPGDv1AntiCorrMax, FirstOrderUPGDv2NormalNormalized, FirstOrderUPGDv1NormalNormalized, FirstOrderUPGDv2NormalMax, FirstOrderUPGDv1NormalMax
from core.optim.weight.gt.second_order import SecondOrderUPGDv2AntiCorrNormalized, SecondOrderUPGDv1AntiCorrNormalized, SecondOrderUPGDv2AntiCorrMax, SecondOrderUPGDv1AntiCorrMax, SecondOrderUPGDv2NormalNormalized, SecondOrderUPGDv1NormalNormalized, SecondOrderUPGDv2NormalMax, SecondOrderUPGDv1NormalMax


class UPGDv2LearnerFOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2AntiCorrNormalized
        name = "upgdv2_anticorr_normalized_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv2LearnerSOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv2AntiCorrNormalized
        name = "upgdv2_anticorr_normalized_so"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv1LearnerFOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1AntiCorrNormalized
        name = "upgdv1_anticorr_normalized_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv1LearnerSOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv1AntiCorrNormalized
        name = "upgdv1_anticorr_normalized_so"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv2LearnerFOAntiCorrMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2AntiCorrMax
        name = "upgdv2_anticorr_max_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv2LearnerSOAntiCorrMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv2AntiCorrMax
        name = "upgdv2_anticorr_max_so"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv1LearnerFOAntiCorrMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1AntiCorrMax
        name = "upgdv1_anticorr_max_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv1LearnerSOAntiCorrMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv1AntiCorrMax
        name = "upgdv1_anticorr_max_so"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)


class UPGDv2LearnerFONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2NormalNormalized
        name = "upgdv2_normal_normalized_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv2LearnerSONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv2NormalNormalized
        name = "upgdv2_normal_normalized_so"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv1LearnerFONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1NormalNormalized
        name = "upgdv1_normal_normalized_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv1LearnerSONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv1NormalNormalized
        name = "upgdv1_normal_normalized_so"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv2LearnerFONormalMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2NormalMax
        name = "upgdv2_normal_max_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv2LearnerSONormalMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv2NormalMax
        name = "upgdv2_normal_max_so"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv1LearnerFONormalMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1NormalMax
        name = "upgdv1_normal_max_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv1LearnerSONormalMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv1NormalMax
        name = "upgdv1_normal_max_so"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)