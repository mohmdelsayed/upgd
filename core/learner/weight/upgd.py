from core.learner.learner import Learner
from core.optim.weight.gt.first_order import FirstOrderUPGDv2AntiCorrNormalized, FirstOrderUPGDv1AntiCorrNormalized, FirstOrderUPGDv2AntiCorrMax, FirstOrderUPGDv1AntiCorrMax, FirstOrderUPGDv2NormalNormalized, FirstOrderUPGDv1NormalNormalized, FirstOrderUPGDv2NormalMax, FirstOrderUPGDv1NormalMax
from core.optim.weight.gt.second_order import SecondOrderUPGDv2AntiCorrNormalized, SecondOrderUPGDv1AntiCorrNormalized, SecondOrderUPGDv2AntiCorrMax, SecondOrderUPGDv1AntiCorrMax, SecondOrderUPGDv2NormalNormalized, SecondOrderUPGDv1NormalNormalized, SecondOrderUPGDv2NormalMax, SecondOrderUPGDv1NormalMax


class UPGDv2LearnerFOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2AntiCorrNormalized
        name = "upgd_v2_fo_anti_corr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv2LearnerSOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv2AntiCorrNormalized
        name = "upgd_v2_so_anti_corr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv1LearnerFOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1AntiCorrNormalized
        name = "upgd_v1_fo_anti_corr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv1LearnerSOAntiCorrNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv1AntiCorrNormalized
        name = "upgd_v1_so_anti_corr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv2LearnerFOAntiCorrMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2AntiCorrMax
        name = "upgd_v2_fo_anti_corr_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv2LearnerSOAntiCorrMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv2AntiCorrMax
        name = "upgd_v2_so_anti_corr_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv1LearnerFOAntiCorrMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1AntiCorrMax
        name = "upgd_v1_fo_anti_corr_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv1LearnerSOAntiCorrMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv1AntiCorrMax
        name = "upgd_v1_so_anti_corr_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)


class UPGDv2LearnerFONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2NormalNormalized
        name = "upgd_v2_fo_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv2LearnerSONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv2NormalNormalized
        name = "upgd_v2_so_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv1LearnerFONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1NormalNormalized
        name = "upgd_v1_fo_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv1LearnerSONormalNormalized(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv1NormalNormalized
        name = "upgd_v1_so_normal_normalized"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv2LearnerFONormalMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv2NormalMax
        name = "upgd_v2_fo_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv2LearnerSONormalMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv2NormalMax
        name = "upgd_v2_so_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)

class UPGDv1LearnerFONormalMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = FirstOrderUPGDv1NormalMax
        name = "upgd_v1_fo_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs)

class UPGDv1LearnerSONormalMax(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SecondOrderUPGDv1NormalMax
        name = "upgd_v1_so_normal_max"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)