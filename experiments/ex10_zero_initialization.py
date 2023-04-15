from core.grid_search import GridSearch
from core.learner.weight.upgd import (
    UPGDv2LearnerFONormalNormalized,
    UPGDv1LearnerFONormalNormalized,
    UPGDv2LearnerFONormalMax,
    UPGDv1LearnerFONormalMax,
)
from core.learner.feature.upgd import (
    FeatureUPGDv2LearnerFONormalNormalized,
    FeatureUPGDv1LearnerFONormalNormalized,
    FeatureUPGDv2LearnerFONormalMax,
    FeatureUPGDv1LearnerFONormalMax,
)
from core.learner.sgd import SGDLearner
from core.learner.pgd import PGDLearner
from core.learner.shrink_and_perturb import ShrinkandPerturbLearner
from core.network.fcn_relu import FullyConnectedReLU, FullyConnectedReLUGates
from core.runner import Runner
from core.run.run_zero_init import RunWithZeroInit
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex10_zero_initialization"
task = tasks[exp_name]()

ups_weight_grids = GridSearch(
        seed=[i for i in range(0, 20)],
        lr=[10 ** -i for i in range(1, 4)],
        beta_utility=[0.999, 0.9999],
        sigma=[0.001, 0.01, 0.1],
        network=[FullyConnectedReLU()],
        n_samples=[1000000],
    )

ups_feature_grids = GridSearch(
        seed=[i for i in range(0, 20)],
        lr=[10 ** -i for i in range(1, 4)],
        beta_utility=[0.999, 0.9999],
        sigma=[0.001, 0.01, 0.1],
        network=[FullyConnectedReLUGates()],
        n_samples=[1000000],
    )

pgd_grids = GridSearch(
               seed=[i for i in range(0, 20)],
               lr=[10 ** -i for i in range(1, 4)],
               sigma=[0.0005, 0.005, 0.05],
               network=[FullyConnectedReLU()],
               n_samples=[1000000],
    )

sp_grids = GridSearch(
               seed=[i for i in range(0, 20)],
                lr=[10 ** -i for i in range(1, 4)],
                sigma=[0.0005, 0.005, 0.05],
                decay=[0.0001, 0.001, 0.01],
                network=[FullyConnectedReLU()],
                n_samples=[1000000],
    )

sgd_grid = GridSearch(
               seed=[i for i in range(0, 20)],
               lr=[10 ** -i for i in range(1, 5)],
               network=[FullyConnectedReLU()],
               n_samples=[1000000],
    )

grids = [ups_feature_grids for _ in range(4)] + [ups_weight_grids for _ in range(4)] + [sgd_grid] + [pgd_grids] + [sp_grids]

learners = [
    FeatureUPGDv2LearnerFONormalNormalized(),
    FeatureUPGDv1LearnerFONormalNormalized(),
    FeatureUPGDv2LearnerFONormalMax(),
    FeatureUPGDv1LearnerFONormalMax(),
    UPGDv2LearnerFONormalNormalized(),
    UPGDv1LearnerFONormalNormalized(),
    UPGDv2LearnerFONormalMax(),
    UPGDv1LearnerFONormalMax(),
    SGDLearner(),
    PGDLearner(),
    ShrinkandPerturbLearner(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(RunWithZeroInit, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")