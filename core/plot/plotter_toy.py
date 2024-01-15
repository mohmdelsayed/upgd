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
    def __init__(self, best_runs_path, metric, task_length=200, n_aggregate_tasks=20):
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
                #   'tab:red',
                  'tab:purple',
                  'tab:brown',
                  'black',
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
                    elif self.metric == "loss":
                        configuration_list.append(data["losses"])
                    else:
                        raise Exception("metric must be loss or accuracy")
                    learner_name = data["learner"]

            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))
            x = [self.n_aggregate_tasks * i for i in range(len(mean_list))]
            plt.plot(x, mean_list, label=learner_name, color=color, linewidth=2)
            plt.fill_between(x, mean_list - std_list, mean_list + std_list, alpha=0.1, color=color)
        plt.ylim(bottom=0.05, top=0.21)
        plt.xlabel(f"Task Number", fontsize=22)
        if self.metric == "accuracy":
            plt.ylabel("Averaged Online Accuracy", fontsize=22)
        elif self.metric == "loss":
            plt.ylabel("Averaged Online Loss", fontsize=22)
        else:
            raise Exception("metric must be loss or accuracy")
        plt.savefig("ss.pdf", bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":

    # best_runs1 = [
    #         'logs_cedar/ex4_changing_average/upgd_v2_fo_normal_max/fully_connected_linear/lr_0.01_beta_utility_0.9_sigma_0.1',
    #         'logs_cedar/ex4_changing_average/sgd/fully_connected_linear/lr_0.01',
    #         'logs_cedar/ex4_changing_average/pgd/fully_connected_linear/lr_0.01_sigma_0.05',
    #         'logs_cedar/ex4_changing_average/upgd_v1_fo_normal_max/fully_connected_linear/lr_0.01_beta_utility_0.999_sigma_0.1',
    #         'logs_cedar/ex4_changing_average/shrink_and_perturb/fully_connected_linear/lr_0.01_decay_0.1_sigma_0.5',
    #         'logs_cedar/ex4_changing_average/sgd/linear_layer/lr_0.01',
    #         'logs_cedar/ex4_changing_average/upgd_v2_so_normal_max/fully_connected_linear/lr_0.01_beta_utility_0.0_sigma_0.1',
    #         'logs_cedar/ex4_changing_average/upgd_v1_so_normal_max/fully_connected_linear/lr_0.01_beta_utility_0.0_sigma_0.1',


    #         # 'logs_cedar/ex4_changing_average/upgd_v2_fo_normal_normalized/fully_connected_linear/lr_0.01_beta_utility_0.9_sigma_0.1',
    #         # 'logs_cedar/ex4_changing_average/sgd/fully_connected_linear/lr_0.01',
    #         # 'logs_cedar/ex4_changing_average/pgd/fully_connected_linear/lr_0.01_sigma_0.05',
    #         # 'logs_cedar/ex4_changing_average/upgd_v1_fo_normal_normalized/fully_connected_linear/lr_0.01_beta_utility_0.999_sigma_0.1',
    #         # 'logs_cedar/ex4_changing_average/shrink_and_perturb/fully_connected_linear/lr_0.01_decay_0.1_sigma_0.5',
    #         # 'logs_cedar/ex4_changing_average/sgd/linear_layer/lr_0.01',
    #         # 'logs_cedar/ex4_changing_average/upgd_v2_so_normal_normalized/fully_connected_linear/lr_0.01_beta_utility_0.9_sigma_0.1',    
    #         # 'logs_cedar/ex4_changing_average/upgd_v1_so_normal_normalized/fully_connected_linear/lr_0.01_beta_utility_0.0_sigma_0.1',
    #     ]

    # best_runs1 = [
    #         # 'logs_cedar/ex4_changing_average/feature_upgd_v2_fo_normal_max/fully_connected_linear_gates/lr_0.01_beta_utility_0.9_sigma_0.0001',
    #         # 'logs_cedar/ex4_changing_average/sgd/fully_connected_linear/lr_0.01',
    #         # 'logs_cedar/ex4_changing_average/pgd/fully_connected_linear/lr_0.01_sigma_0.05',
    #         # 'logs_cedar/ex4_changing_average/feature_upgd_v1_fo_normal_max/fully_connected_linear_gates/lr_0.01_beta_utility_0.99_sigma_0.1',
    #         # 'logs_cedar/ex4_changing_average/shrink_and_perturb/fully_connected_linear/lr_0.01_decay_0.1_sigma_0.5',
    #         # 'logs_cedar/ex4_changing_average/sgd/linear_layer/lr_0.01',
    #         # 'logs_cedar/ex4_changing_average/feature_upgd_v2_so_normal_max/fully_connected_linear_gates/lr_0.01_beta_utility_0.0_sigma_0.01',        
    #         # 'logs_cedar/ex4_changing_average/feature_upgd_v1_so_normal_max/fully_connected_linear_gates/lr_0.01_beta_utility_0.0_sigma_0.1',

    #         'logs_cedar/ex4_changing_average/feature_upgd_v2_fo_normal_normalized/fully_connected_linear_gates/lr_0.01_beta_utility_0.0_sigma_0.01',
    #         'logs_cedar/ex4_changing_average/sgd/fully_connected_linear/lr_0.01',
    #         'logs_cedar/ex4_changing_average/pgd/fully_connected_linear/lr_0.01_sigma_0.05',
    #         'logs_cedar/ex4_changing_average/feature_upgd_v1_fo_normal_normalized/fully_connected_linear_gates/lr_0.01_beta_utility_0.999_sigma_0.1',
    #         'logs_cedar/ex4_changing_average/shrink_and_perturb/fully_connected_linear/lr_0.01_decay_0.1_sigma_0.5',
    #         'logs_cedar/ex4_changing_average/sgd/linear_layer/lr_0.01',
    #         'logs_cedar/ex4_changing_average/feature_upgd_v2_so_normal_normalized/fully_connected_linear_gates/lr_0.01_beta_utility_0.0_sigma_0.0001',
    #         'logs_cedar/ex4_changing_average/feature_upgd_v1_so_normal_normalized/fully_connected_linear_gates/lr_0.01_beta_utility_0.0_sigma_0.1',
    #     ]


    # best_runs1 = [

    #     # 'logs_cedar/ex3_permuted_average/upgd_v2_fo_normal_max/fully_connected_linear/lr_0.01_beta_utility_0.9_sigma_0.0001',
    #     # 'logs_cedar/ex3_permuted_average/sgd/fully_connected_linear/lr_0.01',
    #     # 'logs_cedar/ex3_permuted_average/pgd/fully_connected_linear/lr_0.01_sigma_0.05',
    #     # 'logs_cedar/ex3_permuted_average/upgd_v1_fo_normal_max/fully_connected_linear/lr_0.01_beta_utility_0.999_sigma_0.1',
    #     # 'logs_cedar/ex3_permuted_average/shrink_and_perturb/fully_connected_linear/lr_0.01_decay_0.1_sigma_0.5',
    #     # 'logs_cedar/ex3_permuted_average/sgd/linear_layer/lr_0.01',        
    #     # 'logs_cedar/ex3_permuted_average/upgd_v2_so_normal_max/fully_connected_linear/lr_0.01_beta_utility_0.0_sigma_0.1',
    #     # 'logs_cedar/ex3_permuted_average/upgd_v1_so_normal_max/fully_connected_linear/lr_0.01_beta_utility_0.0_sigma_0.1',

    #     'logs_cedar/ex3_permuted_average/upgd_v2_fo_normal_normalized/fully_connected_linear/lr_0.01_beta_utility_0.0_sigma_0.001',
    #     'logs_cedar/ex3_permuted_average/sgd/fully_connected_linear/lr_0.01',
    #     'logs_cedar/ex3_permuted_average/pgd/fully_connected_linear/lr_0.01_sigma_0.05',
    #     'logs_cedar/ex3_permuted_average/upgd_v1_fo_normal_normalized/fully_connected_linear/lr_0.01_beta_utility_0.999_sigma_0.1',
    #     'logs_cedar/ex3_permuted_average/shrink_and_perturb/fully_connected_linear/lr_0.01_decay_0.1_sigma_0.5',
    #     'logs_cedar/ex3_permuted_average/sgd/linear_layer/lr_0.01',        
    #     'logs_cedar/ex3_permuted_average/upgd_v2_so_normal_normalized/fully_connected_linear/lr_0.01_beta_utility_0.0_sigma_0.001',    
    #     'logs_cedar/ex3_permuted_average/upgd_v1_so_normal_normalized/fully_connected_linear/lr_0.01_beta_utility_0.0_sigma_0.1',
        
    # ]

    # feature-wise:
    best_runs1 = [
        # 'logs_cedar/ex3_permuted_average/feature_upgd_v2_fo_normal_max/fully_connected_linear_gates/lr_0.01_beta_utility_0.999_sigma_0.1',
        # 'logs_cedar/ex3_permuted_average/sgd/fully_connected_linear/lr_0.01',
        # 'logs_cedar/ex3_permuted_average/pgd/fully_connected_linear/lr_0.01_sigma_0.05',
        # 'logs_cedar/ex3_permuted_average/feature_upgd_v1_fo_normal_max/fully_connected_linear_gates/lr_0.01_beta_utility_0.99_sigma_0.1',
        # 'logs_cedar/ex3_permuted_average/shrink_and_perturb/fully_connected_linear/lr_0.01_decay_0.1_sigma_0.5',
        # 'logs_cedar/ex3_permuted_average/sgd/linear_layer/lr_0.01',
        # 'logs_cedar/ex3_permuted_average/feature_upgd_v2_so_normal_max/fully_connected_linear_gates/lr_0.01_beta_utility_0.0_sigma_0.01',
        # 'logs_cedar/ex3_permuted_average/feature_upgd_v1_so_normal_max/fully_connected_linear_gates/lr_0.01_beta_utility_0.0_sigma_0.1',

        'logs_cedar/ex3_permuted_average/feature_upgd_v2_fo_normal_normalized/fully_connected_linear_gates/lr_0.01_beta_utility_0.0_sigma_0.01',
        'logs_cedar/ex3_permuted_average/sgd/fully_connected_linear/lr_0.01',
        'logs_cedar/ex3_permuted_average/pgd/fully_connected_linear/lr_0.01_sigma_0.05',
        'logs_cedar/ex3_permuted_average/feature_upgd_v1_fo_normal_normalized/fully_connected_linear_gates/lr_0.01_beta_utility_0.999_sigma_0.1',
        'logs_cedar/ex3_permuted_average/shrink_and_perturb/fully_connected_linear/lr_0.01_decay_0.1_sigma_0.5',
        'logs_cedar/ex3_permuted_average/sgd/linear_layer/lr_0.01',
        'logs_cedar/ex3_permuted_average/feature_upgd_v2_so_normal_normalized/fully_connected_linear_gates/lr_0.01_beta_utility_0.0_sigma_0.01',
        'logs_cedar/ex3_permuted_average/feature_upgd_v1_so_normal_normalized/fully_connected_linear_gates/lr_0.01_beta_utility_0.0_sigma_0.1',
    ]

    print(best_runs1)
    plotter = Plotter(best_runs1, metric="loss")
    plotter.plot()