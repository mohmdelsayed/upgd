import json
import matplotlib.pyplot as plt
from core.best_run import BestRun
import os
import numpy as np
import matplotlib
matplotlib.rc('figure', figsize=(8, 4))
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 14})

class Plotter:
    def __init__(self, best_runs_path, metric, task_length=2500, n_aggregate_tasks=1):
        self.best_runs_path = best_runs_path
        self.task_length = task_length
        self.avg_interval = task_length * n_aggregate_tasks
        self.n_aggregate_tasks = n_aggregate_tasks
        self.metric = metric

    def plot(self):
        colors = [
                  'tab:blue',
                  'tab:orange',
                  'tab:green',
                  'tab:red',
                  'tab:purple',
                  'tab:brown',
                  'tab:pink',
                  'tab:gray',
                  'tab:olive',
                  'tab:cyan',
                  ]
        for color, subdir in zip(colors, self.best_runs_path):
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    if self.metric == "accuracy":
                        configuration_list.append(data["accuracies"])
                        # configuration_list.append(data["held_out_accuracies"])
                    elif self.metric == "loss":
                        configuration_list.append(data["losses"])
                        # configuration_list.append(data["held_out_averages"])
                    else:
                        raise Exception("metric must be loss or accuracy")
                    learner_name = data["learner"]

            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))
            x = [self.n_aggregate_tasks * i for i in range(len(mean_list))]
            plt.plot(x, mean_list, label=learner_name, color=color)
            plt.fill_between(x, mean_list - std_list, mean_list + std_list, alpha=0.1, color=color)
            plt.legend()

        plt.xlabel(f"Task Number", fontsize=20)
        if self.metric == "accuracy":
            plt.ylabel("Accuracy", fontsize=20)
        elif self.metric == "loss":
            plt.ylabel("Loss", fontsize=20)
        else:
            raise Exception("metric must be loss or accuracy")
        plt.savefig("ss.pdf", bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":

    # best_runs1 = BestRun("ex7_label_permuted_mnist_offline", "area", "fully_connected_relu", ["sgd", "upgd_fo_global"]).get_best_run(measure="losses")
    best_runs1 = BestRun("ex7_label_permuted_mnist_offline", "area", "fully_connected_relu", ["sgd"]).get_best_run(measure="accuracies")

    print(best_runs1)
    # plotter = Plotter(best_runs1, metric="loss")
    plotter = Plotter(best_runs1, metric="accuracy")
    plotter.plot()
