from core.grid_search import GridSearch
from core.learner.adam import AdamLearner
from core.network.fcn_relu import FullyConnectedReLU
from core.runner import Runner
from core.run.run_restarts import RunWithRestarts
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex6_input_permuted_mnist_restarts"
task = tasks[exp_name]()
n_steps = 1000000
n_seeds = 20

adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[10 ** -i for i in range(1, 6)],
               eps=[1e-8],
               network=[FullyConnectedReLU()],
               n_samples=[n_steps],
    )

grids = [adam_grid]

learners = [
    AdamLearner(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(RunWithRestarts, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")