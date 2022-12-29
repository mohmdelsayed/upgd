from core.grid_search import GridSearch
from core.learner.upgd import UPGDv2NormalizedLearnerFO, UPGDv2NormalizedLearnerSO
from core.learner.search import SearchLearnerNormalFO, SearchLearnerNormalSO, SearchLearnerAntiCorrFO, SearchLearnerAntiCorrSO
from core.learner.sgd import SGDLearner
from core.learner.anti_pgd import AntiPGDLearner
from core.learner.pgd import PGDLearner
from core.task.summer_with_sign_change import SummerWithSignChange
from core.network.fcn_tanh import FullyConnectedTanh
from core.runner import Runner
from core.run import Run
from core.utils import create_script_generator, create_script_runner


task = SummerWithSignChange()

grids = [
    GridSearch(
        seed=[i for i in range(0, 2)],
        # lr=[10 ** -i for i in range(0, 3)],
        lr=[0.01],
        beta_utility=[0.0],
        temp=[1.0],
        sigma=[1.0],
        network=[FullyConnectedTanh()],
        n_samples=[20000],
    ),
    GridSearch(
        seed=[i for i in range(0, 2)],
        # lr=[10 ** -i for i in range(0, 3)],
        lr=[0.01],
        beta_utility=[0.0],
        temp=[1.0],
        sigma=[1.0],
        network=[FullyConnectedTanh()],
        n_samples=[20000],
    ),
    GridSearch(
        seed=[i for i in range(0, 2)],
        # lr=[10 ** -i for i in range(0, 3)],
        lr=[0.01],
        beta_utility=[0.0],
        temp=[1.0],
        sigma=[1.0],
        network=[FullyConnectedTanh()],
        n_samples=[20000],
    ),
    GridSearch(
        seed=[i for i in range(0, 2)],
        # lr=[10 ** -i for i in range(0, 3)],
        lr=[0.01],
        beta_utility=[0.0],
        temp=[1.0],
        sigma=[1.0],
        network=[FullyConnectedTanh()],
        n_samples=[20000],
    ),
    GridSearch(
        seed=[i for i in range(0, 2)],
        # lr=[10 ** -i for i in range(0, 3)],
        lr=[0.01],
        beta_utility=[0.0],
        temp=[1.0],
        sigma=[1.0],
        network=[FullyConnectedTanh()],
        n_samples=[20000],
    ),
    GridSearch(
        seed=[i for i in range(0, 2)],
        # lr=[10 ** -i for i in range(0, 3)],
        lr=[0.01],
        beta_utility=[0.0],
        temp=[1.0],
        sigma=[1.0],
        network=[FullyConnectedTanh()],
        n_samples=[20000],
    ),
    GridSearch(seed=[i for i in range(0, 2)],
            #    lr=[10 ** -i for i in range(0, 3)],
               lr=[0.01],
               network=[FullyConnectedTanh()],
               n_samples=[20000],
    ),
    GridSearch(seed=[i for i in range(0, 2)],
            #    lr=[10 ** -i for i in range(0, 3)],
               lr=[0.01],
               network=[FullyConnectedTanh()],
               n_samples=[20000],
    ),
    GridSearch(seed=[i for i in range(0, 2)],
            #    lr=[10 ** -i for i in range(0, 3)],
               lr=[0.01],
               network=[FullyConnectedTanh()],
               n_samples=[20000],
    ),
]

learners = [
    UPGDv2NormalizedLearnerFO(FullyConnectedTanh(), dict()),
    UPGDv2NormalizedLearnerSO(FullyConnectedTanh(), dict()),
    SearchLearnerNormalFO(FullyConnectedTanh(), dict()),
    SearchLearnerNormalSO(FullyConnectedTanh(), dict()),
    SearchLearnerAntiCorrFO(FullyConnectedTanh(), dict()),
    SearchLearnerAntiCorrSO(FullyConnectedTanh(), dict()),
    SGDLearner(FullyConnectedTanh(), dict()),
    AntiPGDLearner(FullyConnectedTanh(), dict()),
    PGDLearner(FullyConnectedTanh(), dict()),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, task, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{task.name}")
    create_script_runner(f"generated_cmds/{task.name}")