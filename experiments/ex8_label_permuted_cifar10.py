# from core.grid_search import GridSearch
# from core.learner.online_ewc_plus import OnlineEWCLearnerPlus
# from core.learner.synaptic_intelligence import SynapticIntelligenceLearner
# from core.network.fcn_relu import ConvolutionalNetworkReLU
# from core.runner import Runner
# from core.run.run import Run
# from core.utils import create_script_generator, create_script_runner, tasks

# exp_name = "ex8_label_permuted_cifar10"
# task = tasks[exp_name]()

# rwalk_grid = GridSearch(
#                seed=[i for i in range(0, 5)],
#                lr=[0.01, 0.001, 0.0001],
#                lamda=[0.01, 0.1, 10.0, 100.0],
#                beta_weight=[0.9, 0.99, 0.999, 0.9999],
#                beta_importance=[0.9, 0.99, 0.999, 0.9999],
#                network=[ConvolutionalNetworkReLU()],
#                n_samples=[1000000],
#     )

# grids = [rwalk_grid] + [rwalk_grid]

# learners = [
#     OnlineEWCLearnerPlus(),
#     SynapticIntelligenceLearner(),
# ]


# for learner, grid in zip(learners, grids):
#     runner = Runner(Run, learner, grid, exp_name, learner.name)
#     runner.write_cmd("generated_cmds")
#     create_script_generator(f"generated_cmds/{exp_name}", exp_name)
#     create_script_runner(f"generated_cmds/{exp_name}")


from core.grid_search import GridSearch
from core.learner.weight.upgd import (
    FirstOrderGlobalUPGDLearner,
    FirstOrderNonprotectingGlobalUPGDLearner,
)
from core.network.fcn_relu import ConvolutionalNetworkReLU
from core.runner import Runner
from core.run.run import Run
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex8_label_permuted_cifar10"
task = tasks[exp_name]()

n_seeds = 20
n_steps = 1000000


# 'logs/ex8_label_permuted_cifar10/upgd_fo_global/convolutional_network_relu/lr_0.01_beta_utility_0.999_sigma_0.01_weight_decay_0.0',

upgd_grid_no_decay = GridSearch(
               seed=[i for i in range(5, n_seeds)],
               lr=[0.01],
               beta_utility=[0.999],
               sigma=[0.01],
               weight_decay=[0.0],
               network=[ConvolutionalNetworkReLU()],
               n_samples=[n_steps],
    )


# 'logs/ex8_label_permuted_cifar10/upgd_fo_global/convolutional_network_relu/lr_0.01_beta_utility_0.999_sigma_0.0_weight_decay_0.0001',

upgd_grid_no_perturb = GridSearch(
               seed=[i for i in range(5, n_seeds)],
               lr=[0.01],
               beta_utility=[0.999],
               sigma=[0.0],
               weight_decay=[0.0001],
               network=[ConvolutionalNetworkReLU()],
               n_samples=[n_steps],
    )


# 'logs/ex8_label_permuted_cifar10/upgd_nonprotecting_fo_global/convolutional_network_relu/lr_0.001_beta_utility_0.9999_sigma_0.001_weight_decay_0.0'

n_upgd_grid_no_decay = GridSearch(
               seed=[i for i in range(5, n_seeds)],
               lr=[0.001],
               beta_utility=[0.9999],
               sigma=[0.001],
               weight_decay=[0.0],
               network=[ConvolutionalNetworkReLU()],
               n_samples=[n_steps],
    )

# 'logs/ex8_label_permuted_cifar10/upgd_nonprotecting_fo_global/convolutional_network_relu/lr_0.01_beta_utility_0.999_sigma_0.0_weight_decay_0.001'

n_upgd_grid_no_perturb = GridSearch(
               seed=[i for i in range(5, n_seeds)],
               lr=[0.01],
               beta_utility=[0.999],
               sigma=[0.0],
               weight_decay=[0.001],
               network=[ConvolutionalNetworkReLU()],
               n_samples=[n_steps],
    )

grids = [upgd_grid_no_decay, upgd_grid_no_perturb, n_upgd_grid_no_decay, n_upgd_grid_no_perturb]

learners = [
    FirstOrderGlobalUPGDLearner(),
    FirstOrderGlobalUPGDLearner(),
    FirstOrderNonprotectingGlobalUPGDLearner(),
    FirstOrderNonprotectingGlobalUPGDLearner(),
]


for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")