from core.learner.learner import Learner
from core.optim.online_ewc import OnlineEWC
from core.optim.noisy_online_ewc import NoisyOnlineEWC

class OnlineEWCLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = OnlineEWC
        name = "online_ewc"
        super().__init__(name, network, optimizer, optim_kwargs)

class NoisyOnlineEWCLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = NoisyOnlineEWC
        name = "noisy_online_ewc"
        super().__init__(name, network, optimizer, optim_kwargs)