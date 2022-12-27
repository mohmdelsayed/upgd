from core.grid_search import GridSearch
from core.learner.upgd import UPGDv2NormalizedLearner
from core.learner.sgd import SGDLearner
from core.task.summer_with_sign_change import SummerWithSignChange
from core.network.fully_connected_tanh import FullyConnectedTanh
from core.network.fully_connected_relu import FullyConnectedReLU
from core.runner import Runner
from core.run import Run


task = SummerWithSignChange()

grids = [
    GridSearch(
        seed=[i for i in range(0, 10)],
        lr=[10 ** -i for i in range(0, 3)],
        beta_utility=[9 * 10 ** -i for i in range(1, 3)],
        temp=[1.0],
        sigma=[1.0],
        network=[FullyConnectedTanh()],
        n_samples=[5],
    ),
    GridSearch(seed=[i for i in range(0, 10)],
               lr=[10 ** -i for i in range(0, 3)],
               network=[FullyConnectedTanh()],
               n_samples=[5],
    ),
]

learners = [
    UPGDv2NormalizedLearner(FullyConnectedTanh(), dict()),
    SGDLearner(FullyConnectedTanh(), dict()),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, task, learner.name)
    runner.write_cmd("generated_cmds")
