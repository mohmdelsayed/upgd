from core.learner.learner import Learner
from core.optim.feature.gt.first_order import FirstOrderUPGDv2AntiCorrNormalized, FirstOrderUPGDv1AntiCorrNormalized, FirstOrderUPGDv1NormalNormalized, FirstOrderUPGDv2NormalNormalized

class FeatureUPGDv2LearnerFOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2AntiCorrNormalized
        name = "feature_upgdv2_fo_anticorr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureUPGDv1LearnerFOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1AntiCorrNormalized
        name = "feature_upgdv1_fo_anticorr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureUPGDv2LearnerFONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2NormalNormalized
        name = "feature_upgdv2_fo_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureUPGDv1LearnerFONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1NormalNormalized
        name = "feature_upgdv1_fo_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)