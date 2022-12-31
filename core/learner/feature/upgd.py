from core.learner.learner import Learner
from core.optim.feature.gt.first_order import FirstOrderUPGDv2AntiCorrNormalized, FirstOrderUPGDv1AntiCorrNormalized

class FeatureUPGDv2LearnerFONormalized(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderUPGDv2AntiCorrNormalized
        name = "feature_upgdv2_fo_anticorr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureUPGDv1LearnerFONormalized(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderUPGDv1AntiCorrNormalized
        name = "feature_upgdv1_fo_anticorr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)