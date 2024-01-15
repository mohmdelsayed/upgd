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

from core.network.fcn_relu import FullyConnectedReLUWithHooks
from core.runner import Runner
from core.run.run_stats import RunStats
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex7_label_permuted_mnist"
task = tasks[exp_name]()

n_steps = 1000000
n_seeds = 20

# 'logs/ex7_label_permuted_mnist/upgd_fo_global/fully_connected_relu/lr_0.01_beta_utility_0.9_sigma_0.001_weight_decay_0.0',

upgd_grid_no_decay = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               beta_utility=[0.9],
               sigma=[0.001],
               weight_decay=[0.0],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )


# 'logs/ex7_label_permuted_mnist/upgd_fo_global/fully_connected_relu/lr_0.01_beta_utility_0.9_sigma_0.0_weight_decay_0.0',

upgd_grid_no_perturb = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               beta_utility=[0.9],
               sigma=[0.0],
               weight_decay=[0.0],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )


# 'logs/ex7_label_permuted_mnist/upgd_nonprotecting_fo_global/fully_connected_relu/lr_0.01_beta_utility_0.99_sigma_0.01_weight_decay_0.0'

n_upgd_grid_no_decay = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               beta_utility=[0.99],
               sigma=[0.01],
               weight_decay=[0.0],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex7_label_permuted_mnist/upgd_nonprotecting_fo_global/fully_connected_relu/lr_0.01_beta_utility_0.999_sigma_0.0_weight_decay_0.0001'

n_upgd_grid_no_perturb = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               beta_utility=[0.999],
               sigma=[0.0],
               weight_decay=[0.0001],
               network=[FullyConnectedReLUWithHooks()],
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
    runner = Runner(RunStats, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")



# # 'logs/ex7_label_permuted_mnist/si_new/fully_connected_relu/lr_0.01_lamda_0.1_beta_weight_0.9_beta_importance_0.9']

# si_grid = GridSearch(
#                 seed=[i for i in range(0, n_seeds)],
#                 lr=[0.01],
#                 beta_weight=[0.9],
#                 beta_importance=[0.9],
#                 lamda=[0.1],
#                 network=[FullyConnectedReLUWithHooks()],
#                 n_samples=[n_steps],
#     )

# # ['logs/ex7_label_permuted_mnist/rwalk/fully_connected_relu/lr_0.01_lamda_0.1_beta_weight_0.9999_beta_importance_0.99',

# ewc_plus_grid = GridSearch(
#                 seed=[i for i in range(0, n_seeds)],
#                 lr=[0.01],
#                 beta_weight=[0.9999],
#                 beta_importance=[0.99],
#                 lamda=[0.1],
#                 network=[FullyConnectedReLUWithHooks()],
#                 n_samples=[n_steps],
#     )
