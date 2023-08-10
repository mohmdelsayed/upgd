from core.learner.learner import Learner
from core.optim.online_ewc_plus import OnlineEWCPlus
from core.optim.noisy_online_ewc_plus import NoisyOnlineEWCPlus

class OnlineEWCLearnerPlus(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = OnlineEWCPlus
        name = "online_ewc_plus"
        super().__init__(name, network, optimizer, optim_kwargs)


class NoisyOnlineEWCLearnerPlus(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = NoisyOnlineEWCPlus
        name = "noisy_online_ewc_plus"
        super().__init__(name, network, optimizer, optim_kwargs)
