from core.learner.learner import Learner
from core.optim.online_ewc import OnlineEWC


class OnlineEWCLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = OnlineEWC
        name = "online_ewc"
        super().__init__(name, network, optimizer, optim_kwargs)