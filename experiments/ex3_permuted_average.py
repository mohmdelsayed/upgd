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

from core.learner.feature.upgd import (
    FeatureUPGDv2LearnerFONormalNormalized,
    FeatureUPGDv2LearnerSONormalNormalized,
    FeatureUPGDv1LearnerFONormalNormalized,
    FeatureUPGDv1LearnerSONormalNormalized,
    FeatureUPGDv2LearnerFONormalMax,
    FeatureUPGDv2LearnerSONormalMax,
    FeatureUPGDv1LearnerFONormalMax,
    FeatureUPGDv1LearnerSONormalMax,
)

from core.learner.sgd import SGDLearner
from core.learner.shrink_and_perturb import ShrinkandPerturbLearner
from core.learner.pgd import PGDLearner
from core.network.fcn_linear import FullyConnectedLinear, LinearLayer, FullyConnectedLinearGates
from core.runner import Runner
from core.run.run import Run
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex3_permuted_average"
task = tasks[exp_name]()

up_grids = GridSearch(
               seed=[i for i in range(0, 10)],
               lr=[10 ** -i for i in range(1, 5)],
               beta_utility=[0.0, 0.9, 0.99, 0.999],
               sigma=[0.0001, 0.001, 0.01, 0.1, 1.0],
               network=[FullyConnectedLinear()],
               n_samples=[1000000],
    )

feature_up_grids = GridSearch(
               seed=[i for i in range(0, 10)],
               lr=[10 ** -i for i in range(1, 5)],
               beta_utility=[0.0, 0.9, 0.99, 0.999],
               sigma=[0.0001, 0.001, 0.01, 0.1, 1.0],
               network=[FullyConnectedLinearGates()],
               n_samples=[1000000],
    )

pgd_grids = GridSearch(
               seed=[i for i in range(0, 30)],
               lr=[10 ** -i for i in range(1, 5)],
               sigma=[0.00005, 0.0005, 0.005, 0.05, 0.5],
               network=[FullyConnectedLinear()],
               n_samples=[1000000],
    )


sgd_grid = GridSearch(
               seed=[i for i in range(0, 30)],
               lr=[10 ** -i for i in range(1, 5)],
               network=[FullyConnectedLinear(), LinearLayer()],
               n_samples=[1000000],
    )

sp_grid = GridSearch(
               seed=[i for i in range(0, 10)],
               lr=[10 ** -i for i in range(1, 5)],
               decay=[0.1, 0.01, 0.001, 0.0001],
               sigma=[0.00005, 0.0005, 0.005, 0.05, 0.5],
               network=[FullyConnectedLinear()],
               n_samples=[1000000],
    )

grids =   [feature_up_grids for _ in range(8)] +  [up_grids for _ in range(8)] + [sgd_grid] + [pgd_grids] + [sp_grid]

learners = [
    FeatureUPGDv2LearnerFONormalNormalized(),
    FeatureUPGDv2LearnerSONormalNormalized(),
    FeatureUPGDv1LearnerFONormalNormalized(),
    FeatureUPGDv1LearnerSONormalNormalized(),
    FeatureUPGDv2LearnerFONormalMax(),
    FeatureUPGDv2LearnerSONormalMax(),
    FeatureUPGDv1LearnerFONormalMax(),
    FeatureUPGDv1LearnerSONormalMax(),
    UPGDv2LearnerFONormalNormalized(),
    UPGDv2LearnerSONormalNormalized(),
    UPGDv1LearnerFONormalNormalized(),
    UPGDv1LearnerSONormalNormalized(),
    UPGDv2LearnerFONormalMax(),
    UPGDv2LearnerSONormalMax(),
    UPGDv1LearnerFONormalMax(),
    UPGDv1LearnerSONormalMax(),
    SGDLearner(),
    PGDLearner(),
    ShrinkandPerturbLearner(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")