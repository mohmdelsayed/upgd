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

exp_name = "ex6_input_permuted_mnist"
task = tasks[exp_name]()

n_steps = 1000000
n_seeds = 20

upgd1_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               beta_utility=[0.9],
               sigma=[0.1],
               weight_decay=[0.01],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/upgd_v1_fo_normal_max/fully_connected_relu/lr_0.001_beta_utility_0.9_sigma_0.1_weight_decay_0.01',

upgd2_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               beta_utility=[0.9999],
               sigma=[0.1],
               weight_decay=[0.01],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/upgd_v2_fo_normal_max/fully_connected_relu/lr_0.01_beta_utility_0.9999_sigma_0.1_weight_decay_0.01',

pgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               sigma=[0.05],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/pgd/fully_connected_relu/lr_0.001_sigma_0.05',

sgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               weight_decay=[0.001],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/sgd/fully_connected_relu/lr_0.001_weight_decay_0.001',

sp_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               sigma=[0.05],
               decay=[0.01],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/shrink_and_perturb/fully_connected_relu/lr_0.001_sigma_0.05_decay_0.01',

adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.0001],
               weight_decay=[0.0],
               beta1=[0.0],
               beta2=[0.99],
               damping=[1e-8],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/adam/fully_connected_relu/lr_0.0001_weight_decay_0.0_beta1_0.0_beta2_0.99_damping_1e-08',

ewc_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               beta_weight=[0.9999],
               beta_fisher=[0.999],
               lamda=[1.0],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/online_ewc/fully_connected_relu/lr_0.001_lamda_1.0_beta_weight_0.9999_beta_fisher_0.999'

noisy_ewc_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                beta_weight=[0.9999],
                beta_fisher=[0.9999],
                lamda=[1.0],
                sigma=[0.01],
                network=[FullyConnectedReLUWithHooks()],
                n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/noisy_online_ewc/fully_connected_relu/lr_0.001_lamda_1.0_beta_weight_0.9999_beta_fisher_0.9999_sigma_0.01',

mas_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                beta_weight=[0.9999],
                beta_fisher=[0.999],
                lamda=[0.1],
                network=[FullyConnectedReLUWithHooks()],
                n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/mas/fully_connected_relu/lr_0.001_lamda_0.1_beta_weight_0.9999_beta_fisher_0.999',

noisy_mas_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                beta_weight=[0.9999],
                beta_fisher=[0.9999],
                lamda=[1.0],
                sigma=[0.01],
                network=[FullyConnectedReLUWithHooks()],
                n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/noisy_mas/fully_connected_relu/lr_0.001_lamda_1.0_beta_weight_0.9999_beta_fisher_0.9999_sigma_0.01',

si_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                beta_weight=[0.99],
                beta_importance=[0.99],
                lamda=[10.0],
                network=[FullyConnectedReLUWithHooks()],
                n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/si/fully_connected_relu/lr_0.001_lamda_10.0_beta_weight_0.99_beta_importance_0.99',

noisy_si_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                beta_weight=[0.99],
                beta_importance=[0.99],
                lamda=[10.0],
                sigma=[0.01],
                network=[FullyConnectedReLUWithHooks()],
                n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/noisy_si/fully_connected_relu/lr_0.001_lamda_10.0_beta_weight_0.99_beta_importance_0.99_sigma_0.01',

bgd_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                mean_eta=[100.0],
                std_init=[0.001],
                beta=[0.9],
                network=[FullyConnectedReLUWithHooks()],
                n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/bgd/fully_connected_relu/mean_eta_100.0_std_init_0.001_beta_0.9',

ewc_plus_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                beta_weight=[0.9999],
                beta_importance=[0.99],
                lamda=[0.1],
                network=[FullyConnectedReLUWithHooks()],
                n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/online_ewc_plus/fully_connected_relu/lr_0.001_lamda_0.1_beta_weight_0.9999_beta_importance_0.99',

noisy_ewc_plus_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                beta_weight=[0.9999],
                beta_importance=[0.999],
                lamda=[0.1],
                sigma=[0.01],
                network=[FullyConnectedReLUWithHooks()],
                n_samples=[n_steps],
    )

# 'logs/ex6_input_permuted_mnist/noisy_online_ewc_plus/fully_connected_relu/lr_0.001_lamda_0.1_beta_weight_0.9999_beta_importance_0.999_sigma_0.01',

grids = [
         upgd1_grid,
         upgd2_grid,
         pgd_grid,
         sgd_grid,
         sp_grid,
         adam_grid,
         ewc_grid,
         noisy_ewc_grid,
         mas_grid,
         noisy_mas_grid,
         si_grid,
         noisy_si_grid,
         bgd_grid,
         ewc_plus_grid,
         noisy_ewc_plus_grid
]

learners = [
    FirstOrderNonprotectingGlobalUPGDLearner(),
    FirstOrderGlobalUPGDLearner(),
    PGDLearner(),
    SGDLearner(),
    ShrinkandPerturbLearner(),
    AdamLearner(),
    OnlineEWCLearner(),
    NoisyOnlineEWCLearner(),
    MASLearner(),
    NoisyMASLearner(),
    SynapticIntelligenceLearner(),
    NoisySynapticIntelligenceLearner(),
    BGDLearner(),
    OnlineEWCLearnerPlus(),
    NoisyOnlineEWCLearnerPlus(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(RunStats, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")