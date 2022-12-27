from core.grid_search import GridSearch
from core.learner.sgd_with_hesscale import SGDwithHesScaleLearner
from core.learner.sgd import SGDLearner
from core.task.utility_task import UtilityTask
from core.network.fully_connected_tanh import FullyConnectedTanh
from core.runner import Runner
from core.run_utility import RunUtility

task = UtilityTask()

grids = [
    GridSearch(seed=[i for i in range(0, 30)],
               lr=[10 ** -i for i in range(0, 1)],
               network=[FullyConnectedTanh()],
               n_samples=[5],
    ),
]

learners = [
    SGDLearner(FullyConnectedTanh(), dict()),
]

for learner, grid in zip(learners, grids):
    runner = Runner(RunUtility, learner, grid, task, learner.name)
    runner.write_cmd("generated_cmds")
