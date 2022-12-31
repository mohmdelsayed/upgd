from core.grid_search import GridSearch
from core.learner.feature.upgd import FeatureUPGDv2Learner
from core.learner.upgd import UPGDv2LearnerFONormalized
from core.learner.search import SearchLearnerAntiCorrFONormalized
from core.learner.sgd import SGDLearner
from core.task.summer_with_sign_change import SummerWithSignChange
from core.network.fcn_tanh import FullyConnectedTanhGates
from core.network.fcn_relu import FullyConnectedReLUGates
from core.network.fcn_leakyrelu import FullyConnectedLeakyReLUGates
from core.runner import Runner
from core.run import Run
from core.utils import create_script_generator, create_script_runner

task = SummerWithSignChange(n_inputs=50)

gt_grids = GridSearch(
        seed=[0],
        lr=[0.01],
        beta_utility=[0.0],
        temp=[1.0],
        sigma=[1.0],
        network=[FullyConnectedTanhGates()],
        n_samples=[50000],
    )

sgd_grids = GridSearch(
               seed=[0],
               lr=[0.01],
               network=[FullyConnectedTanhGates()],
               n_samples=[50000],
    )


grids = [
    gt_grids,
    gt_grids,
    sgd_grids,
]

learners = [
    FeatureUPGDv2Learner(FullyConnectedTanhGates(), dict()),
    SearchLearnerAntiCorrFONormalized(FullyConnectedTanhGates(), dict()),
    SGDLearner(FullyConnectedTanhGates(), dict()),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, task, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{task.name}")
    create_script_runner(f"generated_cmds/{task.name}")