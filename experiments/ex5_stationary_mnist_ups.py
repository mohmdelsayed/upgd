from core.grid_search import GridSearch
from core.learner.weight.search import (
    SearchLearnerAntiCorrFONormalized,
    SearchLearnerAntiCorrSONormalized,
    SearchLearnerAntiCorrFOMax,
    SearchLearnerAntiCorrSOMax,
    SearchLearnerNormalFONormalized,
    SearchLearnerNormalSONormalized,
    SearchLearnerNormalFOMax,
    SearchLearnerNormalSOMax,
)

from core.learner.weight.random import RandomSearchLearnerNormal, RandomSearchLearnerAntiCorr

from core.learner.feature.search import (
    FeatureSearchLearnerAntiCorrFONormalized,
    FeatureSearchLearnerAntiCorrSONormalized,
    FeatureSearchLearnerAntiCorrFOMax,
    FeatureSearchLearnerAntiCorrSOMax,
    FeatureSearchLearnerNormalFONormalized,
    FeatureSearchLearnerNormalSONormalized,
    FeatureSearchLearnerNormalFOMax,
    FeatureSearchLearnerNormalSOMax,
)

from core.learner.feature.random import FeatureRandomSearchLearnerNormal, FeatureRandomSearchLearnerAntiCorr

from core.learner.sgd import SGDLearner
from core.learner.anti_pgd import AntiPGDLearner
from core.learner.pgd import PGDLearner
from core.network.fcn_tanh import FullyConnectedTanh, FullyConnectedTanhGates
from core.network.fcn_relu import FullyConnectedReLU, FullyConnectedReLUGates
from core.runner import Runner
from core.run.run import Run
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex5_stationary_mnist"
task = tasks[exp_name]()

ups_weight_grids = GridSearch(
        seed=[i for i in range(0, 20)],
        lr=[1.0],
        beta_utility=[0.0, 0.9, 0.99, 0.999, 0.9999],
        sigma=[0.0001, 0.001, 0.01, 0.1, 1.0],
        network=[FullyConnectedTanh(), FullyConnectedReLU()],
        n_samples=[1000000],
    )

ups_feature_grids = GridSearch(
        seed=[i for i in range(0, 20)],
        lr=[1.0],
        beta_utility=[0.0, 0.9, 0.99, 0.999, 0.9999],
        sigma=[0.0001, 0.001, 0.01, 0.1, 1.0],
        network=[FullyConnectedTanhGates(), FullyConnectedReLUGates()],
        n_samples=[1000000],
    )


random_weight_grids = GridSearch(
        seed=[i for i in range(0, 20)],
        lr=[1.0],
        sigma=[0.0001, 0.001, 0.01, 0.1, 1.0],
        network=[FullyConnectedTanh(), FullyConnectedReLU()],
        n_samples=[1000000],
    )

random_feature_grids = GridSearch(
        seed=[i for i in range(0, 20)],
        lr=[1.0],
        sigma=[0.0001, 0.001, 0.01, 0.1, 1.0],
        network=[FullyConnectedTanhGates(), FullyConnectedReLUGates()],
        n_samples=[1000000],
    )


grids = [ups_weight_grids for _ in range(8)] + [ups_weight_grids for _ in range(8)] +  [random_weight_grids for _ in range(2)] +  [random_feature_grids for _ in range(2)]

learners = [
    SearchLearnerAntiCorrFONormalized(),
    SearchLearnerAntiCorrSONormalized(),
    SearchLearnerAntiCorrFOMax(),
    SearchLearnerAntiCorrSOMax(),
    SearchLearnerNormalFONormalized(),
    SearchLearnerNormalSONormalized(),
    SearchLearnerNormalFOMax(),
    SearchLearnerNormalSOMax(),

    FeatureSearchLearnerAntiCorrFONormalized(),
    FeatureSearchLearnerAntiCorrSONormalized(),
    FeatureSearchLearnerAntiCorrFOMax(),
    FeatureSearchLearnerAntiCorrSOMax(),
    FeatureSearchLearnerNormalFONormalized(),
    FeatureSearchLearnerNormalSONormalized(),
    FeatureSearchLearnerNormalFOMax(),
    FeatureSearchLearnerNormalSOMax(),

    RandomSearchLearnerNormal(),
    RandomSearchLearnerAntiCorr(),

    FeatureRandomSearchLearnerNormal(),
    FeatureRandomSearchLearnerAntiCorr(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")