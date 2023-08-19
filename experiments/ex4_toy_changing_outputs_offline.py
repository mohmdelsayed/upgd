from core.grid_search import GridSearch


from core.network.fcn_relu import TwoHeadedNetwork
from core.learner.sgd import SGDLearner
from core.runner import Runner
from core.run.run_offline import RunOffline
from core.utils import create_script_generator, create_script_runner, tasks

exp_name = "ex4_toy_changing_outputs_offline"
task = tasks[exp_name]()

sgd_grid = GridSearch(
               seed=[i for i in range(0, 20)],
               lr=[0.01],
               network=[TwoHeadedNetwork()],
               n_samples=[10000],
    )

grids = [sgd_grid]

learners = [
    SGDLearner(),
]

for learner, grid in zip(learners, grids):
    runner = Runner(RunOffline, learner, grid, exp_name, learner.name)
    runner.write_cmd("generated_cmds")
    create_script_generator(f"generated_cmds/{exp_name}", exp_name)
    create_script_runner(f"generated_cmds/{exp_name}")