from core.learner.learner import Learner
from core.optim.mas import MAS
from core.optim.noisy_mas import NoisyMAS

class MASLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = MAS
        name = "mas"
        super().__init__(name, network, optimizer, optim_kwargs)

class NoisyMASLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = NoisyMAS
        name = "noisy_mas"
        super().__init__(name, network, optimizer, optim_kwargs)