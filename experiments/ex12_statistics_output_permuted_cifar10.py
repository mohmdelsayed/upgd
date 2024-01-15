from core.grid_search import GridSearch
from core.learner.weight.upgd import FirstOrderGlobalUPGDLearner, FirstOrderNonprotectingGlobalUPGDLearner
from core.learner.sgd import SGDLearner
from core.learner.pgd import PGDLearner
from core.learner.shrink_and_perturb import ShrinkandPerturbLearner
from core.learner.adam import AdamLearner
from core.learner.bgd import BGDLearner
from core.learner.online_ewc import OnlineEWCLearner, NoisyOnlineEWCLearner
from core.learner.online_ewc_plus import OnlineEWCLearnerPlus, NoisyOnlineEWCLearnerPlus
from core.learner.synaptic_intelligence import SynapticIntelligenceLearner, NoisySynapticIntelligenceLearner 
from core.learner.mas import MASLearner, NoisyMASLearner

from core.network.fcn_relu import ConvolutionalNetworkReLUWithHooks
from core.runner import Runner
from core.run.run_stats import RunStats
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex8_label_permuted_cifar10"
task = tasks[exp_name]()

n_steps = 1000000
n_seeds = 20

# 'logs/ex9_label_permuted_cifar10/online_ewc/convolutional_network_relu/lr_0.01_lamda_10.0_beta_weight_0.999_beta_fisher_0.9999',

ewc_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               beta_weight=[0.999],
               beta_fisher=[0.9999],
               lamda=[10.0],
               network=[ConvolutionalNetworkReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex9_label_permuted_cifar10/mas/convolutional_network_relu/lr_0.01_lamda_10.0_beta_weight_0.999_beta_fisher_0.9999',

mas_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.01],
                beta_weight=[0.999],
                beta_fisher=[0.9999],
                lamda=[10.0],
                network=[ConvolutionalNetworkReLUWithHooks()],
                n_samples=[n_steps],
    )

# 'logs/ex8_label_permuted_cifar10/si_new/convolutional_network_relu/lr_0.001_lamda_0.01_beta_weight_0.99_beta_importance_0.99',

si_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                beta_weight=[0.99],
                beta_importance=[0.99],
                lamda=[0.01],
                network=[ConvolutionalNetworkReLUWithHooks()],
                n_samples=[n_steps],
    )

# 'logs/ex8_label_permuted_cifar10/rwalk/convolutional_network_relu/lr_0.001_lamda_10.0_beta_weight_0.999_beta_importance_0.9',

ewc_plus_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                beta_weight=[0.999],
                beta_importance=[0.9],
                lamda=[10.0],
                network=[ConvolutionalNetworkReLUWithHooks()],
                n_samples=[n_steps],
    )

grids = [
        ewc_grid,
        mas_grid,
        si_grid,
        ewc_plus_grid,
]

learners = [
    OnlineEWCLearner(),
    MASLearner(),
    SynapticIntelligenceLearner(),
    OnlineEWCLearnerPlus(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(RunStats, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")



# # 'logs/ex8_label_permuted_cifar10/si_new/convolutional_network_relu/lr_0.001_lamda_0.01_beta_weight_0.99_beta_importance_0.99'

# si_grid = GridSearch(
#                 seed=[i for i in range(0, n_seeds)],
#                 lr=[0.001],
#                 beta_weight=[0.99],
#                 beta_importance=[0.99],
#                 lamda=[0.01],
#                 network=[ConvolutionalNetworkReLUWithHooks()],
#                 n_samples=[n_steps],
#     )

# # 'logs/ex8_label_permuted_cifar10/rwalk/convolutional_network_relu/lr_0.001_lamda_10.0_beta_weight_0.9999_beta_importance_0.9',

# ewc_plus_grid = GridSearch(
#                 seed=[i for i in range(0, n_seeds)],
#                 lr=[0.001],
#                 beta_weight=[0.9999],
#                 beta_importance=[0.9],
#                 lamda=[10.0],
#                 network=[ConvolutionalNetworkReLUWithHooks()],
#                 n_samples=[n_steps],
#     )
