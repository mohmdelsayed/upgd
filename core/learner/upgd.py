from core.learner.learner import Learner
from core.optim.gt.first_order import FirstOrderUPGDv2AntiCorrNormalized, FirstOrderUPGDv1AntiCorrNormalized
from core.optim.gt.second_order import SecondOrderUPGDv2AntiCorrNormalized, SecondOrderUPGDv1AntiCorrNormalized


class UPGDv2LearnerFO(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderUPGDv2AntiCorrNormalized
        name = "upgdv2_normalized_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv2LearnerSO(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = SecondOrderUPGDv2AntiCorrNormalized
        name = "upgdv2_normalized_so"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv1LearnerFO(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderUPGDv1AntiCorrNormalized
        name = "upgdv1_normalized_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv1LearnerSO(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = SecondOrderUPGDv1AntiCorrNormalized
        name = "upgdv1_normalized_so"
        super().__init__(name, network, optimizer, optim_kwargs)
