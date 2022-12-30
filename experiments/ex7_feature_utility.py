from core.grid_search import GridSearch
from core.learner.sgd import SGDLearner
from core.task.utility_task import UtilityTask
from core.network.fcn_tanh import FullyConnectedTanhGates
from core.runner import Runner
from core.run_utility_feature import FeatureRunUtility
from core.utils import create_script_generator, create_script_runner

task = UtilityTask()

grids = [
    GridSearch(seed=[i for i in range(0, 100)],
               lr=[10**-2],
               network=[FullyConnectedTanhGates()],
               n_samples=[3000],
    ),
]

learners = [
    SGDLearner(FullyConnectedTanhGates(), dict()),
]

for learner, grid in zip(learners, grids):
    runner = Runner(FeatureRunUtility, learner, grid, task, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{task.name}")
    create_script_runner(f"generated_cmds/{task.name}")