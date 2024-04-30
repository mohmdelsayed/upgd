from core.grid_search import GridSearch
from core.learner.sgd import SGDLearnerWithHesScale
from core.network.fcn_tanh import SmallFullyConnectedTanh
from core.network.fcn_leakyrelu import SmallFullyConnectedLeakyReLU
from core.network.fcn_relu import SmallFullyConnectedReLU
from core.network.fcn_linear import SmallFullyConnectedLinear
from core.runner import Runner
from core.run.run_utility import RunUtility
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex1_weight_utils"
task = tasks[exp_name]()
n_steps = 2000
n_seeds = 30

grids = [
    GridSearch(seed=[i for i in range(0, n_seeds)],
               lr=[10**-2],
               network=[SmallFullyConnectedTanh(), SmallFullyConnectedReLU(), SmallFullyConnectedLeakyReLU(), SmallFullyConnectedLinear()],
               n_samples=[n_steps],
    ),
]

learners = [
    SGDLearnerWithHesScale(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(RunUtility, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")