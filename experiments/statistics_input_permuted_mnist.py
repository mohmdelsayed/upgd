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

from core.network.fcn_relu import FullyConnectedReLUWithHooks
from core.runner import Runner
from core.run.run_stats import RunStats
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "input_permuted_mnist_stats"
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

upgd2_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.01],
               beta_utility=[0.9999],
               sigma=[0.1],
               weight_decay=[0.01],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

pgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               sigma=[0.05],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

sgd_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               weight_decay=[0.001],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

sp_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               sigma=[0.05],
               decay=[0.01],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

adam_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.0001],
               weight_decay=[0.0],
               beta1=[0.0],
               beta2=[0.99],
               eps=[1e-8],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

ewc_grid = GridSearch(
               seed=[i for i in range(0, n_seeds)],
               lr=[0.001],
               beta_weight=[0.9999],
               beta_fisher=[0.999],
               lamda=[1.0],
               network=[FullyConnectedReLUWithHooks()],
               n_samples=[n_steps],
    )

mas_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                beta_weight=[0.9999],
                beta_fisher=[0.999],
                lamda=[0.1],
                network=[FullyConnectedReLUWithHooks()],
                n_samples=[n_steps],
    )


si_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                beta_weight=[0.99],
                beta_importance=[0.99],
                lamda=[10.0],
                network=[FullyConnectedReLUWithHooks()],
                n_samples=[n_steps],
    )

rwalk_grid = GridSearch(
                seed=[i for i in range(0, n_seeds)],
                lr=[0.001],
                beta_weight=[0.9999],
                beta_importance=[0.99],
                lamda=[0.1],
                network=[FullyConnectedReLUWithHooks()],
                n_samples=[n_steps],
    )

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
        rwalk_grid
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