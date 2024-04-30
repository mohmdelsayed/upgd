from core.learner.learner import Learner
from core.optim.rwalk import RWalk

class RWalkLearner(Learner):
    def __init__(self, network=None, optim_kwargs={}):
        optimizer = RWalk
        name = "rwalk"
        super().__init__(name, network, optimizer, optim_kwargs)
