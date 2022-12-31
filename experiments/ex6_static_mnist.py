from core.grid_search import GridSearch
from core.learner.weight.upgd import UPGDv2LearnerFONormalized, UPGDv2LearnerSONormalized, UPGDv1LearnerFONormalized, UPGDv1LearnerSONormalized
from core.learner.weight.search import SearchLearnerNormalFONormalized, SearchLearnerNormalSONormalized, SearchLearnerAntiCorrFONormalized, SearchLearnerAntiCorrSONormalized
from core.learner.sgd import SGDLearner
from core.learner.anti_pgd import AntiPGDLearner
from core.learner.pgd import PGDLearner
from core.task.static_mnist import StaticMNIST
from core.network.fcn_tanh import FullyConnectedTanh
from core.network.fcn_relu import FullyConnectedReLU
from core.network.fcn_leakyrelu import FullyConnectedLeakyReLU
from core.runner import Runner
from core.run.run import Run
from core.utils import create_script_generator, create_script_runner


task = StaticMNIST(name="ex6_static_mnist")


gt_grids = GridSearch(
        seed=[0],
        lr=[0.1],
        beta_utility=[0.0],
        temp=[1.0],
        sigma=[1.0],
        network=[FullyConnectedTanh()],
        n_samples=[50000],
    )

sgd_grids = GridSearch(
               seed=[0],
               lr=[0.1],
               network=[FullyConnectedTanh()],
               n_samples=[50000],
    )

grids = [
    gt_grids,
    gt_grids,
    gt_grids,
    gt_grids,
    gt_grids,
    gt_grids,
    gt_grids,
    gt_grids,
    sgd_grids,
    sgd_grids,
    sgd_grids,
]

learners = [
    UPGDv1LearnerFONormalized(),
    UPGDv1LearnerSONormalized(),
    UPGDv2LearnerFONormalized(),
    UPGDv2LearnerSONormalized(),
    SearchLearnerNormalFONormalized(),
    SearchLearnerNormalSONormalized(),
    SearchLearnerAntiCorrFONormalized(),
    SearchLearnerAntiCorrSONormalized(),
    SGDLearner(),
    AntiPGDLearner(),
    PGDLearner(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, task, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{task.name}")
    create_script_runner(f"generated_cmds/{task.name}")