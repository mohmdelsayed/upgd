from core.learner.learner import Learner
from core.optim.sgd import SGD


class SGDLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SGD
        name = "sgd"
        super().__init__(name, network, optimizer, optim_kwargs)

class SGDLearnerWithHesScale(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = SGD
        name = "sgd_with_hesscale"
        super().__init__(name, network, optimizer, optim_kwargs, extend=True)