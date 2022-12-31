from core.grid_search import GridSearch
from core.learner.weight.upgd import (
    UPGDv2LearnerFOAntiCorrNormalized,
    UPGDv1LearnerFOAntiCorrNormalized,
    UPGDv2LearnerFOAntiCorrMax,
    UPGDv1LearnerFOAntiCorrMax,
    UPGDv2LearnerFONormalNormalized,
    UPGDv1LearnerFONormalNormalized,
    UPGDv2LearnerFONormalMax,
    UPGDv1LearnerFONormalMax,
)

from core.learner.weight.search import (
    SearchLearnerAntiCorrFONormalized,
    SearchLearnerAntiCorrFOMax,
    SearchLearnerNormalFONormalized,
    SearchLearnerNormalFOMax,
)
from core.learner.feature.upgd import (
    FeatureUPGDv2LearnerFOAntiCorrNormalized,
    FeatureUPGDv1LearnerFOAntiCorrNormalized,
    FeatureUPGDv2LearnerFOAntiCorrMax,
    FeatureUPGDv1LearnerFOAntiCorrMax,
    FeatureUPGDv2LearnerFONormalNormalized,
    FeatureUPGDv1LearnerFONormalNormalized,
    FeatureUPGDv2LearnerFONormalMax,
    FeatureUPGDv1LearnerFONormalMax,    
)
from core.learner.feature.search import (
    FeatureSearchLearnerAntiCorrFONormalized,
    FeatureSearchLearnerAntiCorrFOMax,
    FeatureSearchLearnerNormalFONormalized,
    FeatureSearchLearnerNormalFOMax,
)
from core.learner.sgd import SGDLearner
from core.learner.anti_pgd import AntiPGDLearner
from core.learner.pgd import PGDLearner
from core.network.fcn_tanh import FullyConnectedTanhGates
from core.network.fcn_relu import FullyConnectedReLUGates
from core.network.fcn_leakyrelu import FullyConnectedLeakyReLUGates
from core.runner import Runner
from core.run.run import Run
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex8_feature_train"
task = tasks[exp_name]()

gt_grids = GridSearch(
        seed=[i for i in range(0, 30)],
        lr=[2 ** -i for i in range(1, 9)],
        beta_utility=[0.0, 0.5, 0.9, 0.99, 0.999],
        temp=[1.0],
        sigma=[2.0, 1.0, 0.5, 0.25],
        network=[FullyConnectedTanhGates(), FullyConnectedReLUGates(), FullyConnectedLeakyReLUGates()],
        n_samples=[20000],
    )

sgd_grid = GridSearch(
               seed=[i for i in range(0, 30)],
               lr=[2 ** -i for i in range(1, 9)],
               network=[FullyConnectedTanhGates(), FullyConnectedReLUGates(), FullyConnectedLeakyReLUGates()],
               n_samples=[20000],
    )

pgd_grids = GridSearch(
               seed=[i for i in range(0, 30)],
               lr=[2 ** -i for i in range(1, 9)],
               sigma=[2.0, 1.0, 0.5, 0.25],
               network=[FullyConnectedTanhGates(), FullyConnectedReLUGates(), FullyConnectedLeakyReLUGates()],
               n_samples=[20000],
    )

# gt_grids = GridSearch(
#         seed=[0],
#         lr=[0.01],
#         beta_utility=[0.0],
#         temp=[1.0],
#         sigma=[1.0],
#         network=[FullyConnectedTanhGates(), FullyConnectedReLUGates(), FullyConnectedLeakyReLUGates()],
#         n_samples=[1],
#     )

# sgd_grid = GridSearch(
#         seed=[0],
#         lr=[0.01],
#         network=[FullyConnectedTanhGates(), FullyConnectedReLUGates(), FullyConnectedLeakyReLUGates()],
#         n_samples=[1],
#     )

# pgd_grids = GridSearch(
#         seed=[0],
#         lr=[0.01],
#         sigma=[1.0],
#         network=[FullyConnectedTanhGates(), FullyConnectedReLUGates(), FullyConnectedLeakyReLUGates()],
#         n_samples=[1],
#     )


grids = [gt_grids for _ in range(24)] + [sgd_grid] + [pgd_grids for _ in range(2)] 

learners = [
    FeatureUPGDv2LearnerFOAntiCorrNormalized(),
    FeatureUPGDv1LearnerFOAntiCorrNormalized(),
    FeatureUPGDv2LearnerFOAntiCorrMax(),
    FeatureUPGDv1LearnerFOAntiCorrMax(),
    FeatureUPGDv2LearnerFONormalNormalized(),
    FeatureUPGDv1LearnerFONormalNormalized(),
    FeatureUPGDv2LearnerFONormalMax(),
    FeatureUPGDv1LearnerFONormalMax(),    
    FeatureSearchLearnerAntiCorrFONormalized(),
    FeatureSearchLearnerAntiCorrFOMax(),
    FeatureSearchLearnerNormalFONormalized(),
    FeatureSearchLearnerNormalFOMax(),
    SearchLearnerAntiCorrFONormalized(),
    SearchLearnerAntiCorrFOMax(),
    SearchLearnerNormalFONormalized(),
    SearchLearnerNormalFOMax(),
    UPGDv2LearnerFOAntiCorrNormalized(),
    UPGDv1LearnerFOAntiCorrNormalized(),
    UPGDv2LearnerFOAntiCorrMax(),
    UPGDv1LearnerFOAntiCorrMax(),
    UPGDv2LearnerFONormalNormalized(),
    UPGDv1LearnerFONormalNormalized(),
    UPGDv2LearnerFONormalMax(),
    UPGDv1LearnerFONormalMax(),
    SGDLearner(),
    AntiPGDLearner(),
    PGDLearner(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}")
    create_script_runner(f"generated_cmds/{exp_name}")