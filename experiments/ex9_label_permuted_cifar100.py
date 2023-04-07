from core.grid_search import GridSearch
from core.learner.weight.upgd import (
    UPGDv2LearnerFONormalNormalized,
    UPGDv2LearnerSONormalNormalized,
    UPGDv1LearnerFONormalNormalized,
    UPGDv1LearnerSONormalNormalized,
    UPGDv2LearnerFONormalMax,
    UPGDv2LearnerSONormalMax,
    UPGDv1LearnerFONormalMax,
    UPGDv1LearnerSONormalMax,
)


from core.learner.sgd import SGDLearner
from core.learner.pgd import PGDLearner
from core.learner.shrink_and_perturb import ShrinkandPerturbLearner
from core.network.fcn_relu import ConvolutionalNetworkReLU
from core.runner import Runner
from core.run.run import Run
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex9_label_permuted_cifar100"
task = tasks[exp_name]()

up_grids = GridSearch(
               seed=[i for i in range(0, 10)],
               lr=[10 ** -i for i in range(2, 6)],
               beta_utility=[0.999, 0.9999],
               sigma=[0.01, 0.1, 1.0],
               weight_decay=[0.0, 0.1, 0.01, 0.001],
               network=[ConvolutionalNetworkReLU()],
               n_samples=[1000000],
    )

pgd_grids = GridSearch(
               seed=[i for i in range(0, 10)],
               lr=[10 ** -i for i in range(2, 6)],
               sigma=[0.005, 0.05, 0.5],
               network=[ConvolutionalNetworkReLU()],
               n_samples=[1000000],
    )


sgd_grid = GridSearch(
               seed=[i for i in range(0, 10)],
               lr=[10 ** -i for i in range(2, 6)],
               network=[ConvolutionalNetworkReLU()],
               n_samples=[1000000],
    )

sp_grid = GridSearch(
               seed=[i for i in range(0, 10)],
               lr=[10 ** -i for i in range(2, 6)],
               sigma=[0.005, 0.05, 0.5],
               decay=[0.1, 0.01, 0.001],
               network=[ConvolutionalNetworkReLU()],
               n_samples=[1000000],
    )

grids = [up_grids for _ in range(4)] + [sgd_grid] +  [pgd_grids] + [sp_grid]

learners = [
    UPGDv2LearnerFONormalNormalized(),
    # UPGDv2LearnerSONormalNormalized(),
    UPGDv1LearnerFONormalNormalized(),
    # UPGDv1LearnerSONormalNormalized(),
    UPGDv2LearnerFONormalMax(),
    # UPGDv2LearnerSONormalMax(),
    UPGDv1LearnerFONormalMax(),
    # UPGDv1LearnerSONormalMax(),
    SGDLearner(),
    PGDLearner(),
    ShrinkandPerturbLearner(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")