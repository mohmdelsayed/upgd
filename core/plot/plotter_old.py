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
        # colors = [
        #             'tab:red',
        #             'tab:blue',
        #             'tab:blue',
        #             'tab:blue',
        #             'tab:blue',
        # ]
        # styles = [
        #             '-',
        #             ':',
        #             '--',
        #             '-.',
        #             '-',
        # ]
        what_to_plot = "accuracies"
        # for style, color, subdir in zip(styles, colors, self.best_runs_path):
        for subdir in self.best_runs_path:
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
            xs = [i * 4 for i in range(self.n_tasks)]
            # plt.plot(xs, mean_list, style, label=learner_name, linewidth=2, color=color)
            # plt.fill_between(xs, mean_list - std_list, mean_list + std_list, alpha=0.1, color=color)
            plt.plot(xs, mean_list, label=learner_name, linewidth=2)
            plt.fill_between(xs, mean_list - std_list, mean_list + std_list, alpha=0.1)
        plt.xlabel(f"Task Number", fontsize=24)
        if self.metric == "accuracy":
            plt.ylabel("Average Online Accuracy", fontsize=24)
        elif self.metric == "loss":
            plt.ylabel("Loss")
        else:
            raise Exception("metric must be loss or accuracy")
        # plt.legend()
        # for input-permuted mnist
        # plt.ylim(bottom=0.68, top=0.79)
        # plt.ylim(bottom=0.27, top=0.51)

        # for output-permuted emnist
        plt.ylim(bottom=0.1)

        # for output-permuted imagenet
        # plt.ylim(bottom=0.0)

        # for dead units:
        # plt.ylim(bottom=0.25)
        # plt.ylim(bottom=0.35, top=1.0)
        # plt.ylim(bottom=0.5)

        # for l0 norm:
        # plt.ylim(bottom=0.0)


        # plt.ylim(bottom=.64, top=0.97)

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

    best_runs1 = [
        # 'logs/ex9_label_permuted_cifar10/upgd_v2_fo_normal_max/convolutional_network_relu/lr_0.01_beta_utility_0.999_sigma_0.001_weight_decay_0.0',
        'logs/ex9_label_permuted_cifar10/upgd_v2_fo_normal_normalized/convolutional_network_relu/lr_0.01_beta_utility_0.9999_sigma_0.001_weight_decay_0.0',
        'logs/ex9_label_permuted_cifar10/sgd/convolutional_network_relu/lr_0.01_weight_decay_0.001',
        'logs/ex9_label_permuted_cifar10/pgd/convolutional_network_relu/lr_0.001_sigma_0.005',
        'logs/ex9_label_permuted_cifar10/adam/convolutional_network_relu/lr_0.001_weight_decay_0.01_beta1_0.0_beta2_0.9999_damping_1e-08',
        # 'logs/ex9_label_permuted_cifar10/upgd_v1_fo_normal_max/convolutional_network_relu/lr_0.01_beta_utility_0.99_sigma_0.01_weight_decay_0.001',
        'logs/ex9_label_permuted_cifar10/upgd_v1_fo_normal_normalized/convolutional_network_relu/lr_0.01_beta_utility_0.999_sigma_0.01_weight_decay_0.001', 
        'logs/ex9_label_permuted_cifar10/shrink_and_perturb/convolutional_network_relu/lr_0.01_sigma_0.005_decay_0.001',

        'logs/ex9_label_permuted_cifar10/online_ewc/convolutional_network_relu/lr_0.01_lamda_10.0_beta_weight_0.999_beta_fisher_0.9999',
        'logs/ex9_label_permuted_cifar10/mas/convolutional_network_relu/lr_0.01_lamda_10.0_beta_weight_0.999_beta_fisher_0.9999',
        'logs/ex8_label_permuted_cifar10/si_new/convolutional_network_relu/lr_0.001_lamda_0.01_beta_weight_0.99_beta_importance_0.99',
        'logs/ex8_label_permuted_cifar10/rwalk/convolutional_network_relu/lr_0.001_lamda_10.0_beta_weight_0.999_beta_importance_0.9',
    ]



    # ablation = [
    #         'logs/ex9_label_permuted_cifar10/sgd/convolutional_network_relu/lr_0.001',
    #         'logs/ex9_label_permuted_cifar10/sgd/convolutional_network_relu/lr_0.01_weight_decay_0.001',
    #         'logs/ex9_label_permuted_cifar10/pgd/convolutional_network_relu/lr_0.001_sigma_0.005',
    #         'logs/ex9_label_permuted_cifar10/shrink_and_perturb/convolutional_network_relu/lr_0.01_sigma_0.005_decay_0.001',
    #         'logs/ex9_label_permuted_cifar10/upgd_v2_fo_normal_max/convolutional_network_relu/lr_0.01_beta_utility_0.999_sigma_0.001_weight_decay_0.0',
    # ]


    print(best_runs1)
    plotter = Plotter(best_runs1, metric="accuracy")
    plotter.plot()