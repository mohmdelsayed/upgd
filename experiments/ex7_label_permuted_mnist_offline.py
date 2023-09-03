from core.grid_search import GridSearch


from core.network.fcn_relu import FullyConnectedReLU
from core.learner.sgd import SGDLearner
from core.learner.adam import AdamLearner
from core.learner.weight.upgd import FirstOrderGlobalUPGDLearner
from core.runner import Runner
from core.run.run_offline import RunOffline
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex7_label_permuted_mnist_offline"
task = tasks[exp_name]()

n_steps = 1000000
n_seeds = 20

sgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               weight_decay=[0.0001],
               network=[FullyConnectedReLU()],
               n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_mnist/sgd/fully_connected_relu/lr_0.01_weight_decay_0.0001',

adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.0001],
               weight_decay=[0.1],
               beta1=[0.0],
               beta2=[0.9999],
               damping=[1e-8],
               network=[FullyConnectedReLU()],
               n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_mnist/adam/fully_connected_relu/lr_0.0001_weight_decay_0.1_beta1_0.0_beta2_0.9999_damping_1e-08',

upgd2_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               beta_utility=[0.9],
               sigma=[0.001],
               weight_decay=[0.0],
               network=[FullyConnectedReLU()],
               n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_mnist/upgd_v2_fo_normal_max/fully_connected_relu/lr_0.01_beta_utility_0.9_sigma_0.001_weight_decay_0.0',

grids = [sgd_grid, adam_grid, upgd2_grid]

learners = [
    SGDLearner(),
    AdamLearner(),
    FirstOrderGlobalUPGDLearner(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(RunOffline, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")