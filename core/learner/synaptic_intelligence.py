from core.learner.learner import Learner
from core.optim.synaptic_intelligence import SynapticIntelligence
from core.optim.noisy_synaptic_intelligence import NoisySynapticIntelligence

class SynapticIntelligenceLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SynapticIntelligence
        name = "si_new"
        super().__init__(name, network, optimizer, optim_kwargs)

class NoisySynapticIntelligenceLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = NoisySynapticIntelligence
        name = "noisy_si"
        super().__init__(name, network, optimizer, optim_kwargs)