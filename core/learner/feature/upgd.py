from core.learner.learner import Learner
from core.optim.feature.upgd.first_order import FirstOrderUPGDv2AntiCorrNormalized, FirstOrderUPGDv1AntiCorrNormalized, FirstOrderUPGDv1NormalNormalized, FirstOrderUPGDv2NormalNormalized, FirstOrderUPGDv1NormalMax, FirstOrderUPGDv2NormalMax, FirstOrderUPGDv1AntiCorrMax, FirstOrderUPGDv2AntiCorrMax
from core.optim.feature.upgd.second_order import SecondOrderUPGDv2AntiCorrNormalized, SecondOrderUPGDv1AntiCorrNormalized, SecondOrderUPGDv1NormalNormalized, SecondOrderUPGDv2NormalNormalized, SecondOrderUPGDv1NormalMax, SecondOrderUPGDv2NormalMax, SecondOrderUPGDv1AntiCorrMax, SecondOrderUPGDv2AntiCorrMax
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

class FeatureUPGDv2LearnerSOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv2AntiCorrNormalized
        name = "feature_upgd_v2_so_anti_corr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureUPGDv1LearnerSOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv1AntiCorrNormalized
        name = "feature_upgd_v1_so_anti_corr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureUPGDv2LearnerSONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv2NormalNormalized
        name = "feature_upgd_v2_so_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureUPGDv1LearnerSONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv1NormalNormalized
        name = "feature_upgd_v1_so_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureUPGDv2LearnerSOAntiCorrMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv2AntiCorrMax
        name = "feature_upgd_v2_so_anti_corr_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureUPGDv1LearnerSOAntiCorrMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv1AntiCorrMax
        name = "feature_upgd_v1_so_anti_corr_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureUPGDv2LearnerSONormalMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv2NormalMax
        name = "feature_upgd_v2_so_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class FeatureUPGDv1LearnerSONormalMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv1NormalMax
        name = "feature_upgd_v1_so_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)