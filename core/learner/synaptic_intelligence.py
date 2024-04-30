from core.learner.learner import Learner
from core.optim.synaptic_intelligence import SynapticIntelligence

class SynapticIntelligenceLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SynapticIntelligence
        name = "si"
        super().__init__(name, network, optimizer, optim_kwargs)
