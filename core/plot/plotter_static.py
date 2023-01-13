import json
import matplotlib.pyplot as plt
from core.best_run import BestRun
import os
import numpy as np
import matplotlib
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 12})

class PlotterStatic:
    def __init__(self, best_runs_path, task_name, avg_interval=100):
        self.best_runs_path = best_runs_path
        self.avg_interval = avg_interval
        self.task_name = task_name

    def plot(self):
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    configuration_list.append(data["losses"])
                    learner_name = data["learner"]

            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))
            plt.plot(mean_list, label=learner_name)
            plt.fill_between(range(len(mean_list)), mean_list - std_list, mean_list + std_list, alpha=0.2)
            plt.ylim([0.0, 1.5])
            plt.legend()
        
        plt.xlabel(f"Bin ({self.avg_interval} sample each)")
        plt.ylabel("Loss")
        plt.savefig("avg_losses.pdf", bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    best_runs = BestRun("ex6_static_mnist", "area", "fully_connected_tanh", ["sgd", "upgdv2_normalized_so"]).get_best_run()
    print(best_runs)
    plotter = PlotterStatic(best_runs, task_name="mnist")
    plotter.plot()
