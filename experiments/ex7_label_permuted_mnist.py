from core.grid_search import GridSearch
from core.learner.weight.upgd import (
    FirstOrderGlobalUPGDLearner,
    FirstOrderNonprotectingGlobalUPGDLearner,
)
from core.network.fcn_relu import FullyConnectedReLU
from core.runner import Runner
from core.run.run import Run
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex7_label_permuted_mnist"
task = tasks[exp_name]()

n_seeds = 5
n_steps = 1000000

perturbation_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01, 0.001, 0.0001],
               beta_utility=[0.9, 0.99, 0.999, 0.9999],
               sigma=[0.0],
               weight_decay=[0.0, 0.0001, 0.001, 0.01, 0.1],
               network=[FullyConnectedReLU()],
               n_samples=[n_steps],
    )


decay_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01, 0.001, 0.0001],
               beta_utility=[0.9, 0.99, 0.999, 0.9999],
               sigma=[0.0, 0.0001, 0.001, 0.01, 0.1],
               weight_decay=[0.0],
               network=[FullyConnectedReLU()],
               n_samples=[n_steps],
    )

# grids = [perturbation_grid] + [decay_grid] +  [perturbation_grid] + [decay_grid]
grids = [perturbation_grid] + [perturbation_grid]

learners = [
    FirstOrderGlobalUPGDLearner(),
    # FirstOrderGlobalUPGDLearner(),
    # FirstOrderNonprotectingGlobalUPGDLearner(),
    FirstOrderNonprotectingGlobalUPGDLearner(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")