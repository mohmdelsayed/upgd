from core.grid_search import GridSearch
from core.task.task import Task
from core.learner.learner import Learner
from core.run.run import Run
import os

class Runner:
    def __init__(
        self, run: Run, learner: Learner, grid_search: GridSearch, exp_name: str, file_name: str,
    ):
        self.grid_search = grid_search
        self.exp_name = exp_name
        self.learner = learner
        self.file_name = file_name
        self.run_name = run.name

    def get_combinations(self):
        return self.grid_search.get_permutations()

    def write_cmd(self, directory):
        cmd = ""
        for permutation in self.get_combinations():
            cmd += f"python3 core/run/{self.run_name}.py --task {self.exp_name} --learner {self.learner}"
            keys, values = zip(*permutation.items())
            for key, value in zip(keys, values):
                cmd += f" --{key} {value}"
            cmd += "\n"

        dir = os.path.join(directory, self.exp_name)
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(f"{dir}/{self.file_name}.txt", "w") as f:
            f.write(cmd)
