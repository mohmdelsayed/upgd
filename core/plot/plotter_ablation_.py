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
    def __init__(self, best_runs_path, metric, avg_interval=2):
        self.best_runs_path = best_runs_path
        self.avg_interval = avg_interval
        self.metric = metric

    def plot(self):

        colors = [
                    'tab:orange',
                    'tab:green',
                    'tab:red',
                    'tab:purple',
                    'tab:brown',
                    'tab:blue',
        ]

        styles = [
                    '-',
                    '-',
                    '-',
                    '-',
                    '-',
                    '-',
        ]
        for style, color, subdir in zip(styles, colors, self.best_runs_path):
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

            if not 'convolutional_network_relu_with_hooks' in subdir:
                self.avg_interval = 10000
            else:
                self.avg_interval = 4
            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))
            xs = [i * 4  for i in range(len(mean_list))]
            plt.plot(xs, mean_list, style, label=learner_name, linewidth=2, color=color)
            plt.fill_between(xs, mean_list - std_list, mean_list + std_list, alpha=0.1, color=color)
        plt.xlabel(f"Task Number", fontsize=22)
        if self.metric == "accuracy":
            plt.ylabel("Average Online Accuracy", fontsize=22)
        elif self.metric == "loss":
            plt.ylabel("Loss")
        else:
            raise Exception("metric must be loss or accuracy")


        # plt.ylim(bottom=0.68, top=0.79)
        plt.ylim(bottom=0.1)
        # plt.ylim(bottom=0.0)

        plt.savefig("ss.pdf", bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":
    # ex6_input_permuted_mnist
    # ex7_label_permuted_mnist
    # ex9_label_permuted_mini_imagenet
    # best_runs1 = BestRun("ex6_input_permuted_mnist", "area", "fully_connected_relu", [
    #                                                                                 # "sgd",
    #                                                                                 # "sgd",
    #                                                                                 # "pgd",
    #                                                                                 # "shrink_and_perturb",
    #                                                                                 "upgd_fo_global",
    #                                                                                  ]).get_best_run(measure="accuracies")


    # best_runs1 = [
    #     'logs/ex6_input_permuted_mnist/sgd/fully_connected_relu_with_hooks/lr_0.001_weight_decay_0.001',
    #     'logs/ex6_input_permuted_mnist/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.001_beta_utility_0.999_sigma_0.0_weight_decay_0.0',
    #     'logs/ex6_input_permuted_mnist/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.001_beta_utility_0.999_sigma_0.01_weight_decay_0.0',
    #     'logs/ex6_input_permuted_mnist/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.9999_sigma_0.1_weight_decay_0.01',
    #     # 'logs/ex6_input_permuted_mnist/upgd_nonprotecting_fo_global/fully_connected_relu_with_hooks/lr_0.001_beta_utility_0.9_sigma_0.1_weight_decay_0.01',

    #     # 'logs/ex6_input_permuted_mnist/upgd_nonprotecting_fo_global/fully_connected_relu_with_hooks/lr_0.001_beta_utility_0.999_sigma_0.0_weight_decay_0.001',

    # # 'logs/ex6_input_permuted_mnist/upgd_nonprotecting_fo_global/fully_connected_relu_with_hooks/lr_0.001_beta_utility_0.99_sigma_0.1_weight_decay_0.0',
    # ]


    # best_runs1 = [
    # 'logs/ex7_label_permuted_mnist/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.9_sigma_0.001_weight_decay_0.0',
    # 'logs/ex7_label_permuted_mnist/upgd_nonprotecting_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.9999_sigma_0.01_weight_decay_0.001',

    # 'logs/ex7_label_permuted_mnist/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.9_sigma_0.0_weight_decay_0.0',
    # 'logs/ex7_label_permuted_mnist/upgd_nonprotecting_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.999_sigma_0.0_weight_decay_0.0001',

    # 'logs/ex7_label_permuted_mnist/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.9_sigma_0.001_weight_decay_0.0',
    # 'logs/ex7_label_permuted_mnist/upgd_nonprotecting_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.99_sigma_0.01_weight_decay_0.0',
    # ]

    # best_runs1 = [
    # 'logs/ex9_label_permuted_mini_imagenet/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.9_sigma_0.001_weight_decay_0.0',
    # 'logs/ex9_label_permuted_mini_imagenet/upgd_nonprotecting_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.999_sigma_0.01_weight_decay_0.0001',

    # 'logs/ex9_label_permuted_mini_imagenet/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.9_sigma_0.0_weight_decay_0.0',
    # 'logs/ex9_label_permuted_mini_imagenet/upgd_nonprotecting_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.99_sigma_0.0_weight_decay_0.001',

    # 'logs/ex9_label_permuted_mini_imagenet/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.9_sigma_0.0_weight_decay_0.0',
    # 'logs/ex9_label_permuted_mini_imagenet/upgd_nonprotecting_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.9_sigma_0.01_weight_decay_0.0',
    # ]

    # best_runs2 = ['logs/ex9_label_permuted_mini_imagenet/sgd/fully_connected_relu/lr_0.01']


    # best_runs1 = [
    #     'logs/ex7_label_permuted_mnist/upgd_fo_global/fully_connected_relu/lr_0.01_beta_utility_0.9_sigma_0.0_weight_decay_0.0', # zero perturbation + zero decay
    #     'logs/ex7_label_permuted_mnist/upgd_fo_global/fully_connected_relu/lr_0.01_beta_utility_0.9_sigma_0.0_weight_decay_0.0001', # zero perturbation
    #     'logs/ex7_label_permuted_mnist/upgd_fo_global/fully_connected_relu/lr_0.01_beta_utility_0.9_sigma_0.0001_weight_decay_0.0', # zero decay
    #     'logs/ex7_label_permuted_mnist/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.9_sigma_0.001_weight_decay_0.0',

    # ]


    # best_runs1 = [
    #     'logs/ex9_label_permuted_mini_imagenet/upgd_fo_global/fully_connected_relu/lr_0.01_beta_utility_0.9_sigma_0.0_weight_decay_0.0',
    #     'logs/ex9_label_permuted_mini_imagenet/upgd_fo_global/fully_connected_relu/lr_0.01_beta_utility_0.9_sigma_0.0_weight_decay_0.0001',
    #     'logs/ex9_label_permuted_mini_imagenet/upgd_fo_global/fully_connected_relu/lr_0.01_beta_utility_0.9_sigma_0.0001_weight_decay_0.0',
    #     'logs/ex9_label_permuted_mini_imagenet/upgd_fo_global/fully_connected_relu_with_hooks/lr_0.01_beta_utility_0.9_sigma_0.001_weight_decay_0.0',
    # ]


    best_runs1 = [
    'logs/ex8_label_permuted_cifar10/upgd_fo_global/convolutional_network_relu/lr_0.01_beta_utility_0.999_sigma_0.0_weight_decay_0.0001',
    'logs/ex8_label_permuted_cifar10/upgd_fo_global/convolutional_network_relu/lr_0.01_beta_utility_0.999_sigma_0.01_weight_decay_0.0',
    'logs/ex8_label_permuted_cifar10/upgd_fo_global/convolutional_network_relu/lr_0.01_beta_utility_0.999_sigma_0.0_weight_decay_0.0',
    'logs/ex8_label_permuted_cifar10/sgd/convolutional_network_relu_with_hooks/lr_0.01_weight_decay_0.001',
    'logs/ex8_label_permuted_cifar10/pgd/convolutional_network_relu_with_hooks/lr_0.001_sigma_0.005',
    'logs/ex8_label_permuted_cifar10/upgd_fo_global/convolutional_network_relu_with_hooks/lr_0.01_beta_utility_0.999_sigma_0.001_weight_decay_0.0',
    ]


    print(best_runs1)
    plotter = Plotter(best_runs1, metric="accuracy")
    plotter.plot()