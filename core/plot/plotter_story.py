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

class Plotter:
    def __init__(self, best_runs_path, metric, task_length=5000, n_aggregate_tasks=1):
        self.best_runs_path = best_runs_path
        self.avg_interval = task_length * n_aggregate_tasks
        self.n_aggregate_tasks = n_aggregate_tasks
        self.tasks_to_plot = 200
        self.task_length = task_length
        self.metric = metric

    def plot(self):
        colors = [
                  'tab:blue',
                  'tab:red',
                  'tab:gray',
                  ]
        for color, subdir in zip(colors, self.best_runs_path):
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
            x = [self.n_aggregate_tasks * i for i in range(len(mean_list))]
            plt.plot(x[:self.tasks_to_plot], mean_list[:self.tasks_to_plot], label=learner_name, color=color, linewidth=2.0)
            plt.fill_between(x[:self.tasks_to_plot], mean_list[:self.tasks_to_plot] - std_list[:self.tasks_to_plot], mean_list[:self.tasks_to_plot] + std_list[:self.tasks_to_plot], alpha=0.1)

        plt.xlabel(f"Task Number", fontsize=24)
        if self.metric == "accuracy":
            plt.ylabel("Average Online Accuracy", fontsize=24)
        elif self.metric == "loss":
            plt.ylabel("Loss", fontsize=24)
        else:
            raise Exception("metric must be loss or accuracy")
        
        plt.ylim(0.67, 0.79)
        plt.savefig("lop.pdf", bbox_inches='tight')
        plt.clf()

    def plot_1st_n_tasks(self, n_tasks=2, avg_interval=100,first_time=25000//100, second_time=875000//100):
        colors = [
                  'tab:blue',
                  'tab:red',
                  'tab:gray',
                  ]
        for color, subdir in zip(colors, self.best_runs_path):
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

            all_data_binned = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // avg_interval, avg_interval).mean(axis=-1)
            all_data_mean = all_data_binned.mean(axis=0)
            all_data_std = all_data_binned.std(axis=0)

            mean_list1 = all_data_mean[first_time : first_time + (n_tasks * self.task_length)//avg_interval ]
            std_list1 = all_data_std[first_time : first_time + (n_tasks * self.task_length) // avg_interval] / np.sqrt(len(seeds) * avg_interval)
            mean_list2 = all_data_mean[second_time : second_time + (n_tasks * self.task_length) // avg_interval]
            std_list2 = all_data_std[second_time : second_time + (n_tasks * self.task_length) // avg_interval] / np.sqrt(len(seeds) * avg_interval)

            concat_mean = np.concatenate([mean_list1, mean_list2])
            concat_std = np.concatenate([std_list1, std_list2])
            
            plt.plot(concat_mean, label=learner_name, color=color, alpha=0.95, linewidth=2.0)
            plt.fill_between([i for i in range(len(concat_mean))], concat_mean - concat_std, concat_mean + concat_std, alpha=0.3, color=color)

            plt.xlabel(f"Number of Samples", fontsize=24)
            if self.metric == "accuracy":
                plt.ylabel("Online Accuracy", fontsize=24)
            elif self.metric == "loss":
                plt.ylabel("Online Loss", fontsize=24)
            else:
                raise Exception("metric must be loss or accuracy")
        
        plt.ylim(bottom=0.5, top=0.9)
        plt.savefig("first_n_tasks.pdf", bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":

    best_runs1 = [
                'logs/ex6_input_permuted_mnist/upgd_v2_fo_normal_max/fully_connected_relu/lr_0.01_beta_utility_0.9999_sigma_0.1_weight_decay_0.01',
                'logs/ex6_input_permuted_mnist/adam/fully_connected_relu/lr_0.0001_weight_decay_0.0_beta1_0.0_beta2_0.99_damping_1e-08',
                'logs/ex6_input_permuted_mnist_restarts/adam/fully_connected_relu/lr_0.0001_beta1_0.0_beta2_0.99_damping_1e-08',
]
    print(best_runs1)

    plotter = Plotter(best_runs1, metric="accuracy")
    plotter.plot()
    plotter.plot_1st_n_tasks()
