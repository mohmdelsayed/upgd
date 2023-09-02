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

class Plotter:
    def __init__(self, best_runs_path, metric, avg_interval=200):
        self.best_runs_path = best_runs_path
        self.avg_interval = avg_interval
        self.metric = metric

    def plot(self):
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    if self.metric == "accuracy":
                        configuration_list.append(data["accuracies"])
                    elif self.metric == "loss":
                        configuration_list.append(data["losses"])
                    else:
                        raise Exception("metric must be loss or accuracy")
                    learner_name = data["learner"]

            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))
            plt.plot(mean_list, label=learner_name)
            plt.fill_between(range(len(mean_list)), mean_list - std_list, mean_list + std_list, alpha=0.1)
            plt.legend()
        
        plt.xlabel(f"Number of tasks")
        if self.metric == "accuracy":
            plt.ylabel("Accuracy")
        elif self.metric == "loss":
            plt.ylabel("Loss")
        else:
            raise Exception("metric must be loss or accuracy")
        # plt.ylim(bottom=0.0)
        plt.savefig("ss.pdf", bbox_inches='tight')
        plt.clf()

    def plot_1st_n_tasks(self, n_tasks=5):
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    if self.metric == "accuracy":
                        configuration_list.append(data["accuracies"])
                    elif self.metric == "loss":
                        configuration_list.append(data["losses"])
                    else:
                        raise Exception("metric must be loss or accuracy")
                    learner_name = data["learner"]

            mean_list = np.array(configuration_list).mean(axis=0)[0:n_tasks * self.avg_interval-1]
            std_list = np.array(configuration_list).std(axis=0)[0:n_tasks * self.avg_interval-1] / np.sqrt(len(seeds))
            plt.plot(mean_list, label=learner_name)
            plt.fill_between(range(len(mean_list)), mean_list - std_list, mean_list + std_list, alpha=0.1)
            plt.legend()
        
        plt.xlabel(f"Sample")
        if self.metric == "accuracy":
            plt.ylabel("Accuracy")
        elif self.metric == "loss":
            plt.ylabel("Loss")
        else:
            raise Exception("metric must be loss or accuracy")
        plt.ylim(bottom=0.0, top=1.0)
        plt.savefig("first_n_tasks.pdf", bbox_inches='tight')
        plt.clf()

    def plot_last_n_tasks(self, n_tasks=5):
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    if self.metric == "accuracy":
                        configuration_list.append(data["accuracies"])
                    elif self.metric == "loss":
                        configuration_list.append(data["losses"])
                    else:
                        raise Exception("metric must be loss or accuracy")
                    learner_name = data["learner"]

            all_mean_list = np.array(configuration_list).mean(axis=0)
            mean_list = all_mean_list[(-n_tasks * self.avg_interval):]
            std_list = np.array(configuration_list).std(axis=0)[(-n_tasks * self.avg_interval):] / np.sqrt(len(seeds))
            plt.plot(np.arange(n_tasks * self.avg_interval) + len(np.array(all_mean_list))  - n_tasks * self.avg_interval, mean_list, label=learner_name)
            plt.ylim(bottom=0.0, top=1.0)
            plt.fill_between(np.arange(n_tasks * self.avg_interval) + len(np.array(all_mean_list))  - n_tasks * self.avg_interval, mean_list - std_list, mean_list + std_list, alpha=0.1)
            plt.legend()
        
        plt.xlabel(f"Sample")
        if self.metric == "accuracy":
            plt.ylabel("Accuracy")
        elif self.metric == "loss":
            plt.ylabel("Loss")
        else:
            raise Exception("metric must be loss or accuracy")
        plt.ylim(bottom=0.0, top=1.0)
        plt.savefig("last_n_tasks.pdf", bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    best_runs1 = BestRun("ex3_permuted_average", "area", "fully_connected_linear", ["sgd", "anti_pgd", "pgd"]).get_best_run(measure="losses")
    best_runs2 = BestRun("ex3_permuted_average", "area", "linear_layer", ["sgd"]).get_best_run(measure="losses")

    print(best_runs1+best_runs2)
    plotter = Plotter(best_runs1+best_runs2, metric="loss")
    plotter.plot()
    plotter.plot_1st_n_tasks(n_tasks=5)
    plotter.plot_last_n_tasks(n_tasks=5)