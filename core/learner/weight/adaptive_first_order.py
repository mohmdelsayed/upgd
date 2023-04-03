from core.learner.learner import Learner
from core.optim.weight.upgd.adaptive_first_order import AdaptiveUPGDAntiCorrLayerwiseFO, AdaptiveUPGDNormalLayerwiseFO

class AdaptiveUPGDAntiCorrLayerwiseFOLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaptiveUPGDAntiCorrLayerwiseFO
        name = "adaptive_upgd_v2_fo_anti_corr_layerwise"
        super().__init__(name, network, optimizer, optim_kwargs)

class AdaptiveUPGDNormalLayerwiseFOLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = AdaptiveUPGDNormalLayerwiseFO
        name = "adaptive_upgd_v2_fo_normal_layerwise"
        super().__init__(name, network, optimizer, optim_kwargs)
