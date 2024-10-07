from core.grid_search import GridSearch
from core.learner.weight_upgd import FirstOrderGlobalUPGDLearner, FirstOrderNonprotectingGlobalUPGDLearner
from core.learner.sgd import SGDLearner
from core.learner.pgd import PGDLearner
from core.learner.shrink_and_perturb import ShrinkandPerturbLearner
from core.learner.adam import AdamLearner
from core.learner.ewc import EWCLearner
from core.learner.rwalk import RWalkLearner
from core.learner.synaptic_intelligence import SynapticIntelligenceLearner
from core.learner.mas import MASLearner

from core.network.fcn_relu import ConvolutionalNetworkReLUWithHooks
from core.runner import Runner
from core.run.run_stats import RunStats
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "label_permuted_cifar10_stats"
task = tasks[exp_name]()

n_steps = 1000000
n_seeds = 20

upgd1_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               beta_utility=[0.99],
               sigma=[0.01],
               weight_decay=[0.001],
               network=[ConvolutionalNetworkReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_cifar10/upgd_v1_fo_normal_max/convolutional_network_relu/lr_0.01_beta_utility_0.99_sigma_0.01_weight_decay_0.001',

upgd2_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               beta_utility=[0.999],
               sigma=[0.001],
               weight_decay=[0.0],
               network=[ConvolutionalNetworkReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_cifar10/upgd_v2_fo_normal_max/convolutional_network_relu/lr_0.01_beta_utility_0.999_sigma_0.001_weight_decay_0.0',

pgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               sigma=[0.005],
               network=[ConvolutionalNetworkReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_cifar10/pgd/convolutional_network_relu/lr_0.001_sigma_0.005',

sgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               weight_decay=[0.001],
               network=[ConvolutionalNetworkReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_cifar10/sgd/convolutional_network_relu/lr_0.01_weight_decay_0.001',

sp_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               sigma=[0.005],
               decay=[0.001],
               network=[ConvolutionalNetworkReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_cifar10/shrink_and_perturb/convolutional_network_relu/lr_0.01_sigma_0.005_decay_0.001',

adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               weight_decay=[0.01],
               beta1=[0.0],
               beta2=[0.9999],
               eps=[1e-8],
               network=[ConvolutionalNetworkReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_cifar10/adam/convolutional_network_relu/lr_0.001_weight_decay_0.01_beta1_0.0_beta2_0.9999_eps_1e-08',

ewc_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               beta_weight=[0.999],
               beta_fisher=[0.9999],
               lamda=[10.0],
               network=[ConvolutionalNetworkReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_cifar10/online_ewc/convolutional_network_relu/lr_0.01_lamda_10.0_beta_weight_0.999_beta_fisher_0.9999'

mas_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.01],
                beta_weight=[0.999],
                beta_fisher=[0.9999],
                lamda=[10.0],
                network=[ConvolutionalNetworkReLUWithHooks()],
                n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_cifar10/mas/convolutional_network_relu/lr_0.01_lamda_10.0_beta_weight_0.999_beta_fisher_0.9999',

si_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                beta_weight=[0.999],
                beta_importance=[0.9],
                lamda=[1.0],
                network=[ConvolutionalNetworkReLUWithHooks()],
                n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_cifar10/si/convolutional_network_relu/lr_0.001_lamda_1.0_beta_weight_0.999_beta_importance_0.9',

rwalk_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.01],
                beta_weight=[0.9999],
                beta_importance=[0.999],
                lamda=[0.1],
                network=[ConvolutionalNetworkReLUWithHooks()],
                n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_cifar10/rwalk/convolutional_network_relu/lr_0.01_lamda_0.1_beta_weight_0.9999_beta_importance_0.999',

grids = [
         upgd1_grid,
         upgd2_grid,
         pgd_grid,
         sgd_grid,
         sp_grid,
         adam_grid,
         ewc_grid,
         mas_grid,
         si_grid,
         rwalk_grid,
]

learners = [
    FirstOrderNonprotectingGlobalUPGDLearner(),
    FirstOrderGlobalUPGDLearner(),
    PGDLearner(),
    SGDLearner(),
    ShrinkandPerturbLearner(),
    AdamLearner(),
    EWCLearner(),
    MASLearner(),
    SynapticIntelligenceLearner(),
    RWalkLearner(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(RunStats, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")