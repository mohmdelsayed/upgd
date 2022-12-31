from core.learner.learner import Learner
from core.optim.weight.gt.first_order import FirstOrderUPGDv2AntiCorrNormalized, FirstOrderUPGDv1AntiCorrNormalized, FirstOrderUPGDv2AntiCorrMax, FirstOrderUPGDv1AntiCorrMax
from core.optim.weight.gt.second_order import SecondOrderUPGDv2AntiCorrNormalized, SecondOrderUPGDv1AntiCorrNormalized, SecondOrderUPGDv2AntiCorrMax, SecondOrderUPGDv1AntiCorrMax


class UPGDv2LearnerFONormalized(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderUPGDv2AntiCorrNormalized
        name = "upgdv2_normalized_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv2LearnerSONormalized(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = SecondOrderUPGDv2AntiCorrNormalized
        name = "upgdv2_normalized_so"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv1LearnerFONormalized(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderUPGDv1AntiCorrNormalized
        name = "upgdv1_normalized_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv1LearnerSONormalized(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = SecondOrderUPGDv1AntiCorrNormalized
        name = "upgdv1_normalized_so"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv2LearnerFOMax(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderUPGDv2AntiCorrMax
        name = "upgdv2_max_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv2LearnerSOMax(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = SecondOrderUPGDv2AntiCorrMax
        name = "upgdv2_max_so"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv1LearnerFOMax(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderUPGDv1AntiCorrMax
        name = "upgdv1_max_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv1LearnerSOMax(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = SecondOrderUPGDv1AntiCorrMax
        name = "upgdv1_max_so"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)