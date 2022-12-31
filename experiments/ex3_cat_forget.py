from core.grid_search import GridSearch
from core.learner.weight.upgd import UPGDv2LearnerFONormalized, UPGDv2LearnerSONormalized, UPGDv1LearnerFONormalized, UPGDv1LearnerSONormalized
from core.learner.weight.search import SearchLearnerNormalFONormalized, SearchLearnerNormalSONormalized, SearchLearnerAntiCorrFONormalized, SearchLearnerAntiCorrSONormalized
from core.learner.sgd import SGDLearner
from core.learner.anti_pgd import AntiPGDLearner
from core.learner.pgd import PGDLearner
from core.task.summer_with_sign_change import SummerWithSignChange
from core.network.fcn_linear import FullyConnectedLinear
from core.runner import Runner
from core.run.run import Run
from core.utils import create_script_generator, create_script_runner

task = SummerWithSignChange()

gt_grids = GridSearch(
        seed=[i for i in range(0, 30)],
        lr=[2 ** -i for i in range(0, 8)],
        beta_utility=[0.0, 0.5, 0.9, 0.99, 0.999],
        temp=[1.0, 2.0, 0.5],
        sigma=[1.0, 0.5, 2.0],
        network=[FullyConnectedLinear()],
        n_samples=[20000],
    )

sgd_grids = GridSearch(
               seed=[i for i in range(0, 30)],
               lr=[2 ** -i for i in range(0, 8)],
               network=[FullyConnectedLinear()],
               n_samples=[20000],
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
    UPGDv1LearnerFONormalized(FullyConnectedLinear(), dict()),
    UPGDv1LearnerSONormalized(FullyConnectedLinear(), dict()),
    UPGDv2LearnerFONormalized(FullyConnectedLinear(), dict()),
    UPGDv2LearnerSONormalized(FullyConnectedLinear(), dict()),
    SearchLearnerNormalFONormalized(FullyConnectedLinear(), dict()),
    SearchLearnerNormalSONormalized(FullyConnectedLinear(), dict()),
    SearchLearnerAntiCorrFONormalized(FullyConnectedLinear(), dict()),
    SearchLearnerAntiCorrSONormalized(FullyConnectedLinear(), dict()),
    SGDLearner(FullyConnectedLinear(), dict()),
    AntiPGDLearner(FullyConnectedLinear(), dict()),
    PGDLearner(FullyConnectedLinear(), dict()),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, task, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{task.name}")
    create_script_runner(f"generated_cmds/{task.name}")