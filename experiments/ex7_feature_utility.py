from core.grid_search import GridSearch
from core.learner.sgd import SGDLearnerWithHesScale
from core.network.fcn_tanh import SmallFullyConnectedTanhGates
from core.network.fcn_leakyrelu import SmallFullyConnectedLeakyReLUGates
from core.network.fcn_relu import SmallFullyConnectedReLUGates
from core.runner import Runner
from core.run.run_utility_feature import FeatureRunUtility
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex7_feature_utils"
task = tasks[exp_name]()

grids = [
    GridSearch(seed=[i for i in range(0, 30)],
               lr=[10**-2],
               network=[SmallFullyConnectedTanhGates(), SmallFullyConnectedLeakyReLUGates(), SmallFullyConnectedReLUGates()],
               n_samples=[2000],
    ),
]

learners = [
    SGDLearnerWithHesScale(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(FeatureRunUtility, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")