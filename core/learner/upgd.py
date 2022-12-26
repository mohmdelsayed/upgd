from core.learner.learner import Learner
from core.optim.gt.first_order import FirstOrderUPGDv2AntiCorrNormalized


class UPGDv2NormalizedLearner(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderUPGDv2AntiCorrNormalized
        name = "upgdv2_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)
