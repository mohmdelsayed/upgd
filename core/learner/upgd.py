from core.learner.learner import Learner
from core.optim.gt.first_order import FirstOrderUPGDv2AntiCorrNormalized
from core.optim.gt.second_order import SecondOrderUPGDv2AntiCorrNormalized


class UPGDv2NormalizedLearnerFO(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderUPGDv2AntiCorrNormalized
        name = "upgdv2_normalized_fo"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv2NormalizedLearnerSO(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = SecondOrderUPGDv2AntiCorrNormalized
        name = "upgdv2_normalized_so"
        super().__init__(name, network, optimizer, optim_kwargs)
