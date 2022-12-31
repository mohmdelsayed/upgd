from core.learner.learner import Learner
from core.optim.feature.gt.first_order import FirstOrderUPGDv2AntiCorrNormalized, FirstOrderUPGDv1AntiCorrNormalized, FirstOrderUPGDv1NormalNormalized, FirstOrderUPGDv2NormalNormalized, FirstOrderUPGDv1NormalMax, FirstOrderUPGDv2NormalMax, FirstOrderUPGDv1AntiCorrMax, FirstOrderUPGDv2AntiCorrMax

class FeatureUPGDv2LearnerFOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2AntiCorrNormalized
        name = "feature_upgd_v2_fo_anti_corr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureUPGDv1LearnerFOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1AntiCorrNormalized
        name = "feature_upgd_v1_fo_anti_corr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureUPGDv2LearnerFONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2NormalNormalized
        name = "feature_upgd_v2_fo_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureUPGDv1LearnerFONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1NormalNormalized
        name = "feature_upgd_v1_fo_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureUPGDv2LearnerFOAntiCorrMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2AntiCorrMax
        name = "feature_upgd_v2_fo_anti_corr_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureUPGDv1LearnerFOAntiCorrMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1AntiCorrMax
        name = "feature_upgd_v1_fo_anti_corr_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureUPGDv2LearnerFONormalMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2NormalMax
        name = "feature_upgd_v2_fo_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class FeatureUPGDv1LearnerFONormalMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1NormalMax
        name = "feature_upgd_v1_fo_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs)