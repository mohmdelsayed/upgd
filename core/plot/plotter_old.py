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
    def __init__(self, best_runs_path, metric, avg_interval=10000):
        self.best_runs_path = best_runs_path
        self.avg_interval = avg_interval
        self.metric = metric
        self.n_tasks = 100

    def plot(self):
        colors = [
                    'tab:red',
                    'tab:blue',
                    'tab:blue',
                    'tab:blue',
                    'tab:blue',
        ]
        styles = [
                    '-',
                    ':',
                    '--',
                    '-.',
                    '-',
        ]
        what_to_plot = "accuracies"
        for style, color, subdir in zip(styles, colors, self.best_runs_path):
        # for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    if self.metric == "accuracy":
                        configuration_list.append(data[what_to_plot])
                    elif self.metric == "loss":
                        configuration_list.append(data["losses"])
                    else:
                        raise Exception("metric must be loss or accuracy")
                    learner_name = data["learner"]

            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))
            xs = [i*4  for i in range(self.n_tasks)]
            # plt.plot(xs, mean_list, label=learner_name, linewidth=2)
            # plt.fill_between(xs, mean_list - std_list, mean_list + std_list, alpha=0.1)
            plt.plot(xs, mean_list, style, label=learner_name, linewidth=2, color=color)
            plt.fill_between(xs, mean_list - std_list, mean_list + std_list, alpha=0.1, color=color)
            
        # plt.ylim(bottom=0.68, top=0.79)
        plt.ylim(bottom=0.0)
        plt.title('Ablation on label-permuted mini-ImageNet', x=0.5, y=0.95, fontsize=15)

        plt.xlabel(f"Task Number", fontsize=24)
        if self.metric == "accuracy":
            plt.ylabel("Average Online Accuracy", fontsize=24)
        elif self.metric == "loss":
            plt.ylabel("Loss")
        else:
            raise Exception("metric must be loss or accuracy")
        # plt.legend()
        plt.savefig("ss.pdf", bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":

    # best_runs1 = [
    # # 'logs/ex9_label_permuted_mnist/upgd_v2_fo_normal_max/fully_connected_relu/lr_0.01_beta_utility_0.9_sigma_0.001_weight_decay_0.0',
    # # 'logs/ex9_label_permuted_mnist/upgd_v2_fo_normal_normalized/fully_connected_relu/lr_0.01_beta_utility_0.99_sigma_0.01_weight_decay_0.0',
    # # 'logs/ex9_label_permuted_mnist/feature_upgd_v2_fo_normal_max/fully_connected_relu_gates/lr_0.01_beta_utility_0.9999_sigma_0.01_weight_decay_0.001',
    # 'logs/ex9_label_permuted_mnist/feature_upgd_v2_fo_normal_normalized/fully_connected_relu_gates/lr_0.01_beta_utility_0.9999_sigma_0.01_weight_decay_0.001',
    # 'logs/ex9_label_permuted_mnist/sgd/fully_connected_relu/lr_0.01_weight_decay_0.0001',
    # 'logs/ex9_label_permuted_mnist/pgd/fully_connected_relu/lr_0.01_sigma_0.005',
    # 'logs/ex9_label_permuted_mnist/adam/fully_connected_relu/lr_0.0001_weight_decay_0.1_beta1_0.0_beta2_0.9999_damping_1e-08',
    # # 'logs/ex9_label_permuted_mnist/upgd_v1_fo_normal_max/fully_connected_relu/lr_0.01_beta_utility_0.9999_sigma_0.01_weight_decay_0.001',
    # # 'logs/ex9_label_permuted_mnist/upgd_v1_fo_normal_normalized/fully_connected_relu/lr_0.01_beta_utility_0.999_sigma_0.01_weight_decay_0.001',
    # # 'logs/ex9_label_permuted_mnist/feature_upgd_v1_fo_normal_max/fully_connected_relu_gates/lr_0.01_beta_utility_0.99_sigma_0.1_weight_decay_0.001',
    # 'logs/ex9_label_permuted_mnist/feature_upgd_v1_fo_normal_normalized/fully_connected_relu_gates/lr_0.01_beta_utility_0.9999_sigma_0.1_weight_decay_0.001',
    # 'logs/ex9_label_permuted_mnist/shrink_and_perturb/fully_connected_relu/lr_0.01_sigma_0.005_decay_0.001',

    # 'logs/ex9_label_permuted_mnist/online_ewc/fully_connected_relu/lr_0.01_lamda_1.0_beta_weight_0.999_beta_fisher_0.999',
    # 'logs/ex9_label_permuted_mnist/mas/fully_connected_relu/lr_0.01_lamda_10.0_beta_weight_0.999_beta_fisher_0.999',
    # 'logs/ex7_label_permuted_mnist/si_new/fully_connected_relu/lr_0.01_lamda_0.1_beta_weight_0.9_beta_importance_0.9',
    # 'logs/ex7_label_permuted_mnist/rwalk/fully_connected_relu/lr_0.01_lamda_0.1_beta_weight_0.999_beta_importance_0.9',
    # ]



    # best_runs1 = [
    #     # 'logs/ex6_input_permuted_mnist/upgd_v2_fo_normal_max/fully_connected_relu/lr_0.01_beta_utility_0.9999_sigma_0.1_weight_decay_0.01',
    #     # 'logs/ex6_input_permuted_mnist/upgd_v2_fo_normal_normalized/fully_connected_relu/lr_0.01_beta_utility_0.9999_sigma_0.1_weight_decay_0.01',
    #     # 'logs/ex6_input_permuted_mnist/feature_upgd_v2_fo_normal_max/fully_connected_relu_gates/lr_0.01_beta_utility_0.9999_sigma_0.1_weight_decay_0.01',
    #     'logs/ex6_input_permuted_mnist/feature_upgd_v2_fo_normal_normalized/fully_connected_relu_gates/lr_0.01_beta_utility_0.9999_sigma_0.1_weight_decay_0.01',
    #     'logs/ex6_input_permuted_mnist/sgd/fully_connected_relu/lr_0.001_weight_decay_0.001',
    #     'logs/ex6_input_permuted_mnist/pgd/fully_connected_relu/lr_0.001_sigma_0.05',
    #     'logs/ex6_input_permuted_mnist/adam/fully_connected_relu/lr_0.0001_weight_decay_0.0_beta1_0.0_beta2_0.99_damping_1e-08',
    #     # 'logs/ex6_input_permuted_mnist/upgd_v1_fo_normal_max/fully_connected_relu/lr_0.001_beta_utility_0.9_sigma_0.1_weight_decay_0.01',
    #     # 'logs/ex6_input_permuted_mnist/upgd_v1_fo_normal_normalized/fully_connected_relu/lr_0.001_beta_utility_0.999_sigma_0.1_weight_decay_0.01',
    #     'logs/ex6_input_permuted_mnist/feature_upgd_v1_fo_normal_normalized/fully_connected_relu_gates/lr_0.001_beta_utility_0.999_sigma_0.1_weight_decay_0.01',
    #     # 'logs/ex6_input_permuted_mnist/feature_upgd_v1_fo_normal_max/fully_connected_relu_gates/lr_0.001_beta_utility_0.999_sigma_0.1_weight_decay_0.01',
    #     'logs/ex6_input_permuted_mnist/shrink_and_perturb/fully_connected_relu/lr_0.001_sigma_0.05_decay_0.01',

    #     'logs/ex6_input_permuted_mnist/online_ewc/fully_connected_relu/lr_0.001_lamda_0.001_beta_weight_0.99_beta_fisher_0.9999',
    #     'logs/ex6_input_permuted_mnist/mas/fully_connected_relu/lr_0.001_lamda_0.1_beta_weight_0.999_beta_fisher_0.9999',
    #     'logs/ex6_input_permuted_mnist/si_new/fully_connected_relu/lr_0.001_lamda_0.1_beta_weight_0.999_beta_importance_0.9999',
    #     'logs/ex6_input_permuted_mnist/rwalk/fully_connected_relu/lr_0.001_lamda_10.0_beta_weight_0.999_beta_importance_0.99',
    # ]


    # best_runs1 = [
    #     'logs/ex6_input_permuted_mnist/sgd/fully_connected_relu/lr_0.001',
    #     'logs/ex6_input_permuted_mnist/sgd/fully_connected_relu/lr_0.001_weight_decay_0.001',
    #     'logs/ex6_input_permuted_mnist/pgd/fully_connected_relu/lr_0.001_sigma_0.05',
    #     'logs/ex6_input_permuted_mnist/shrink_and_perturb/fully_connected_relu/lr_0.001_sigma_0.05_decay_0.01',
    #     'logs/ex6_input_permuted_mnist/upgd_v2_fo_normal_max/fully_connected_relu/lr_0.01_beta_utility_0.9999_sigma_0.1_weight_decay_0.01',
    # ]
    # best_runs1 = [
    #     'logs/ex9_label_permuted_mnist/sgd/fully_connected_relu/lr_0.01',
    #     'logs/ex9_label_permuted_mnist/sgd/fully_connected_relu/lr_0.01_weight_decay_0.0001',
    #     'logs/ex9_label_permuted_mnist/pgd/fully_connected_relu/lr_0.01_sigma_0.005',
    #     'logs/ex9_label_permuted_mnist/shrink_and_perturb/fully_connected_relu/lr_0.01_sigma_0.005_decay_0.001',
    #     'logs/ex9_label_permuted_mnist/upgd_v2_fo_normal_max/fully_connected_relu/lr_0.01_beta_utility_0.9_sigma_0.001_weight_decay_0.0',
    # ]

    # best_runs1 = [
    # 'logs/ex9_label_permuted_mini_imagenet/sgd/fully_connected_relu/lr_0.01',
    # 'logs/ex9_label_permuted_mini_imagenet/sgd/fully_connected_relu/lr_0.01_weight_decay_0.001',
    # 'logs/ex9_label_permuted_mini_imagenet/pgd/fully_connected_relu/lr_0.01_sigma_0.005',
    # 'logs/ex9_label_permuted_mini_imagenet/shrink_and_perturb/fully_connected_relu/lr_0.01_sigma_0.005_decay_0.001',
    # 'logs/ex9_label_permuted_mini_imagenet/upgd_v2_fo_normal_max/fully_connected_relu/lr_0.01_beta_utility_0.9_sigma_0.001_weight_decay_0.0',
    # ]
    print(best_runs1)
    plotter = Plotter(best_runs1, metric="accuracy")
    plotter.plot()