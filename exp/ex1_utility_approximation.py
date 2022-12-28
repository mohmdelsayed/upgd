from core.grid_search import GridSearch
from core.learner.sgd_with_hesscale import SGDwithHesScaleLearner
from core.learner.sgd import SGDLearner
from core.task.utility_task import UtilityTask
from core.network.fcn_tanh import SmallFullyConnectedTanh
from core.runner import Runner
from core.run_utility import RunUtility
from core.utils import create_script_generator, create_script_runner

task = UtilityTask()

grids = [
    GridSearch(seed=[i for i in range(0, 30)],
               lr=[10**-2],
               network=[SmallFullyConnectedTanh()],
               n_samples=[3000],
    ),
]

learners = [
    SGDLearner(SmallFullyConnectedTanh(), dict()),
]

for learner, grid in zip(learners, grids):
    runner = Runner(RunUtility, learner, grid, task, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{task.name}")
    create_script_runner(f"generated_cmds/{task.name}")