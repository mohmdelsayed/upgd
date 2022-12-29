import json
import matplotlib.pyplot as plt
from core.best_run import BestRun
import os
import numpy as np

class Plotter:
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
            plt.title(self.task_name)
            plt.fill_between(range(len(mean_list)), mean_list - std_list, mean_list + std_list, alpha=0.1)
            plt.legend()
        
        plt.xlabel(f"Task")
        plt.ylabel("Loss")
        plt.show()

    def plot_1st_n_tasks(self, n_tasks=5):
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    configuration_list.append(data["losses"])
                    learner_name = data["learner"]

            mean_list = np.array(configuration_list).mean(axis=0)[0:n_tasks * self.avg_interval-1]
            std_list = np.array(configuration_list).std(axis=0)[0:n_tasks * self.avg_interval-1] / np.sqrt(len(seeds))
            plt.plot(mean_list, label=learner_name)
            plt.title(f"First {n_tasks} tasks {self.task_name}")
            plt.fill_between(range(len(mean_list)), mean_list - std_list, mean_list + std_list, alpha=0.1)
            plt.legend()
        
        plt.xlabel(f"Time Step")
        plt.ylabel("Loss")
        plt.show()

    def plot_last_n_tasks(self, n_tasks=5):
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    configuration_list.append(data["losses"])
                    learner_name = data["learner"]

            all_mean_list = np.array(configuration_list).mean(axis=0)
            mean_list = all_mean_list[(-n_tasks * self.avg_interval):]
            std_list = np.array(configuration_list).std(axis=0)[(-n_tasks * self.avg_interval):] / np.sqrt(len(seeds))
            plt.plot(np.arange(n_tasks * self.avg_interval) + len(np.array(all_mean_list))  - n_tasks * self.avg_interval, mean_list, label=learner_name)
            plt.title(f"Last {n_tasks} tasks {self.task_name}")
            plt.fill_between(np.arange(n_tasks * self.avg_interval) + len(np.array(all_mean_list))  - n_tasks * self.avg_interval, mean_list - std_list, mean_list + std_list, alpha=0.1)
            plt.legend()
        
        plt.xlabel(f"Time Step")
        plt.ylabel("Loss")
        plt.show()


if __name__ == "__main__":
    # best_runs = BestRun("summer_with_sign_change", "area", "fully_connected_tanh", ["sgd", "upgdv2_normalized"]).get_best_run()
    best_runs = BestRun("summer_with_sign_change", "area", "fully_connected_linear", ["sgd", "upgdv2_normalized_fo", "upgdv2_normalized_so"]).get_best_run()
    # best_runs = BestRun("summer_with_signals_change", "area", "fully_connected_tanh", ["sgd", "upgdv2_normalized"]).get_best_run()
    # best_runs = BestRun("label_permuted_mnist", "area", "fully_connected_tanh", ["sgd", "upgdv2_normalized"]).get_best_run()

    print(best_runs)
    plotter = Plotter(best_runs, task_name="mnist")
    plotter.plot()
    plotter.plot_1st_n_tasks(n_tasks=5)
    plotter.plot_last_n_tasks(n_tasks=5)