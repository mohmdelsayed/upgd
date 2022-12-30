from core.learner.learner import Learner
from core.optim.search.first_order import FeatureSearchAntiCorrNormalized

class FeatureUPGDv2Learner(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FeatureSearchAntiCorrNormalized
        name = "feature_upgdv2"
        super().__init__(name, network, optimizer, optim_kwargs)
