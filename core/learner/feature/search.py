from core.learner.learner import Learner
from core.optim.feature.search.first_order import FirstOrderSearchAntiCorrNormalized

class FeatureSearchLearnerNormalFONormalized(Learner):
    def __init__(self, network, optim_kwargs):
        optimizer = FirstOrderSearchAntiCorrNormalized
        name = "feature_search_fo_anticorr_normalized"
        super().__init__(name, network, optimizer, optim_kwargs)