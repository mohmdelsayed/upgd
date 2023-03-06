from core.grid_search import GridSearch
from core.learner.weight.upgd import (
    UPGDv2LearnerFOAntiCorrNormalized,
    UPGDv2LearnerSOAntiCorrNormalized,
    UPGDv1LearnerFOAntiCorrNormalized,
    UPGDv1LearnerSOAntiCorrNormalized,
    UPGDv2LearnerFOAntiCorrMax,
    UPGDv2LearnerSOAntiCorrMax,
    UPGDv1LearnerFOAntiCorrMax,
    UPGDv1LearnerSOAntiCorrMax,
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
from core.learner.anti_pgd import AntiPGDLearner
from core.learner.pgd import PGDLearner
from core.network.fcn_tanh import FullyConnectedTanh
from core.network.fcn_relu import FullyConnectedReLU
from core.network.fcn_leakyrelu import FullyConnectedLeakyReLU
from core.network.fcn_linear import FullyConnectedLinear, LinearLayer
from core.runner import Runner
from core.run.run import Run
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex8_two_label_change_mnist"
task = tasks[exp_name]()

up_grids = GridSearch(
               seed=[i for i in range(0, 30)],
               lr=[2 ** -i for i in range(1, 7)],
               beta_utility=[0.0],
               temp=[1.0],
               sigma=[1.0],
               network=[FullyConnectedTanh(), FullyConnectedReLU(), FullyConnectedLeakyReLU(), FullyConnectedLinear(), LinearLayer()],
               n_samples=[250000],
               noise_damping=[0, 1],
    )

pgd_grids = GridSearch(
               seed=[i for i in range(0, 30)],
               lr=[2 ** -i for i in range(1, 7)],
               sigma=[1.0],
               network=[FullyConnectedTanh(), FullyConnectedReLU(), FullyConnectedLeakyReLU(), FullyConnectedLinear(), LinearLayer()],
               n_samples=[250000],
    )


sgd_grid = GridSearch(
               seed=[i for i in range(0, 30)],
               lr=[2 ** -i for i in range(1, 7)],
               network=[FullyConnectedTanh(), FullyConnectedReLU(), FullyConnectedLeakyReLU(), FullyConnectedLinear(), LinearLayer()],
               n_samples=[250000],
    )

grids = [up_grids for _ in range(16)] + [sgd_grid] + [pgd_grids for _ in range(2)] 

learners = [
    UPGDv2LearnerFOAntiCorrNormalized(),
    UPGDv2LearnerSOAntiCorrNormalized(),
    UPGDv1LearnerFOAntiCorrNormalized(),
    UPGDv1LearnerSOAntiCorrNormalized(),
    UPGDv2LearnerFOAntiCorrMax(),
    UPGDv2LearnerSOAntiCorrMax(),
    UPGDv1LearnerFOAntiCorrMax(),
    UPGDv1LearnerSOAntiCorrMax(),
    UPGDv2LearnerFONormalNormalized(),
    UPGDv2LearnerSONormalNormalized(),
    UPGDv1LearnerFONormalNormalized(),
    UPGDv1LearnerSONormalNormalized(),
    UPGDv2LearnerFONormalMax(),
    UPGDv2LearnerSONormalMax(),
    UPGDv1LearnerFONormalMax(),
    UPGDv1LearnerSONormalMax(),
    SGDLearner(),
    AntiPGDLearner(),
    PGDLearner(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(Run, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")