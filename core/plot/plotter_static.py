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
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 18})

class PlotterStatic:
    def __init__(self, best_runs_path, task_name, avg_interval=10000):
        self.best_runs_path = best_runs_path
        self.avg_interval = avg_interval
        self.task_name = task_name

    def plot(self):
        colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:purple",
            "tab:brown",
        ]
        for color, subdir in zip(colors, self.best_runs_path):
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    configuration_list.append(data["accuracies"])
                    learner_name = data["learner"]

            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))
            x = [self.avg_interval * i for i in range(len(mean_list))]
            plt.plot(x, mean_list, label=learner_name, color=color, linewidth=2)
            plt.fill_between(x, mean_list - std_list, mean_list + std_list, alpha=0.1, color=color)
            # plt.legend()
            plt.fill_between(range(len(mean_list)), mean_list - std_list, mean_list + std_list, alpha=0.2)
            plt.ylim([0.8, 1.0])
        
        plt.xlabel(f"Number of Samples", fontsize=22)
        plt.ylabel("Averaged Online Accuracy", fontsize=22)
        plt.savefig("avg_losses.pdf", bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":

    best_run = [
    # 'logs_cedar/ex5_stationary_mnist/upgd_v2_fo_normal_max/fully_connected_relu/lr_0.01_beta_utility_0.9999_sigma_0.001',
    # 'logs_cedar/ex5_stationary_mnist/sgd/fully_connected_relu/lr_0.01',
    # 'logs_cedar/ex5_stationary_mnist/pgd/fully_connected_relu/lr_0.01_sigma_0.0005',
    # 'logs_cedar/ex5_stationary_mnist/upgd_v1_fo_normal_max/fully_connected_relu/lr_0.001_beta_utility_0.999_sigma_0.001',
    # 'logs_cedar/ex5_stationary_mnist/shrink_and_perturb/fully_connected_relu/lr_0.01_sigma_0.005_decay_0.001',

    # 'logs_cedar/ex5_stationary_mnist/upgd_v2_fo_normal_normalized/fully_connected_relu/lr_0.01_beta_utility_0.99_sigma_0.001',
    # 'logs_cedar/ex5_stationary_mnist/sgd/fully_connected_relu/lr_0.01',
    # 'logs_cedar/ex5_stationary_mnist/pgd/fully_connected_relu/lr_0.01_sigma_0.0005',
    # 'logs_cedar/ex5_stationary_mnist/upgd_v1_fo_normal_normalized/fully_connected_relu/lr_0.001_beta_utility_0.99_sigma_0.001',
    # 'logs_cedar/ex5_stationary_mnist/shrink_and_perturb/fully_connected_relu/lr_0.01_sigma_0.005_decay_0.001',

    # 'logs_cedar/ex5_stationary_mnist/feature_upgd_v2_fo_normal_max/fully_connected_relu_gates/lr_0.01_beta_utility_0.9999_sigma_0.001',
    # 'logs_cedar/ex5_stationary_mnist/sgd/fully_connected_relu/lr_0.01',
    # 'logs_cedar/ex5_stationary_mnist/pgd/fully_connected_relu/lr_0.01_sigma_0.0005',
    # 'logs_cedar/ex5_stationary_mnist/feature_upgd_v1_fo_normal_max/fully_connected_relu_gates/lr_0.001_beta_utility_0.99_sigma_0.01',
    # 'logs_cedar/ex5_stationary_mnist/shrink_and_perturb/fully_connected_relu/lr_0.01_sigma_0.005_decay_0.001',

    'logs_cedar/ex5_stationary_mnist/feature_upgd_v2_fo_normal_normalized/fully_connected_relu_gates/lr_0.01_beta_utility_0.999_sigma_0.001',
    'logs_cedar/ex5_stationary_mnist/sgd/fully_connected_relu/lr_0.01',
    'logs_cedar/ex5_stationary_mnist/pgd/fully_connected_relu/lr_0.01_sigma_0.0005',
    'logs_cedar/ex5_stationary_mnist/feature_upgd_v1_fo_normal_normalized/fully_connected_relu_gates/lr_0.001_beta_utility_0.99_sigma_0.01',
    'logs_cedar/ex5_stationary_mnist/shrink_and_perturb/fully_connected_relu/lr_0.01_sigma_0.005_decay_0.001',
    ]

    print(best_run)
    plotter = PlotterStatic(best_run, task_name="mnist")
    plotter.plot()
