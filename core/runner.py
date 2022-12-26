from core.grid_search import GridSearch
from core.task.task import Task
from core.learner.learner import Learner
import os

class Runner:
    def __init__(
        self, learner: Learner, grid_search: GridSearch, task: Task, file_name: str
    ):
        self.grid_search = grid_search
        self.task = task
        self.learner = learner
        self.file_name = file_name

    def get_combinations(self):
        return self.grid_search.get_permutations()

    def write_cmd(self, directory):
        cmd = ""
        for permutation in self.get_combinations():
            cmd += f"python3 core/run.py --task {self.task.name} --learner {self.learner}"
            keys, values = zip(*permutation.items())
            for key, value in zip(keys, values):
                cmd += f" --{key} {value}"
            cmd += "\n"

        dir = os.path.join(directory, self.task.name)
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(f"{dir}/{self.file_name}.txt", "w") as f:
            f.write(cmd)
