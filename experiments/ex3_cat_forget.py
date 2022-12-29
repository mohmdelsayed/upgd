from core.grid_search import GridSearch
from core.learner.upgd import UPGDv1LearnerFO, UPGDv1LearnerSO, UPGDv2LearnerFO, UPGDv2LearnerSO
from core.learner.search import SearchLearnerNormalFO, SearchLearnerNormalSO, SearchLearnerAntiCorrFO, SearchLearnerAntiCorrSO
from core.learner.sgd import SGDLearner
from core.learner.anti_pgd import AntiPGDLearner
from core.learner.pgd import PGDLearner
from core.task.summer_with_sign_change import SummerWithSignChange
from core.network.fcn_linear import FullyConnectedLinear
from core.runner import Runner
from core.run import Run
from core.utils import create_script_generator, create_script_runner

task = SummerWithSignChange()

gt_grids = GridSearch(
        seed=[i for i in range(0, 2)],
        # lr=[10 ** -i for i in range(0, 3)],
        lr=[0.01],
        beta_utility=[0.0],
        temp=[1.0],
        sigma=[1.0],
        network=[FullyConnectedLinear()],
        n_samples=[20000],
    )

sgd_grids = GridSearch(seed=[i for i in range(0, 2)],
            #    lr=[10 ** -i for i in range(0, 3)],
               lr=[0.01],
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
    UPGDv1LearnerFO(FullyConnectedLinear(), dict()),
    UPGDv1LearnerSO(FullyConnectedLinear(), dict()),
    UPGDv2LearnerFO(FullyConnectedLinear(), dict()),
    UPGDv2LearnerSO(FullyConnectedLinear(), dict()),
    SearchLearnerNormalFO(FullyConnectedLinear(), dict()),
    SearchLearnerNormalSO(FullyConnectedLinear(), dict()),
    SearchLearnerAntiCorrFO(FullyConnectedLinear(), dict()),
    SearchLearnerAntiCorrSO(FullyConnectedLinear(), dict()),
    SGDLearner(FullyConnectedLinear(), dict()),
    AntiPGDLearner(FullyConnectedLinear(), dict()),
    PGDLearner(FullyConnectedLinear(), dict()),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, task, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{task.name}")
    create_script_runner(f"generated_cmds/{task.name}")