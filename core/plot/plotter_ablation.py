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
        # what_to_plot = "losses"
        what_to_plot = "accuracies"
        # what_to_plot = "plasticity_per_task"
        # what_to_plot = "n_dead_units_per_task"
        # what_to_plot = "grad_l0_per_task"
        # what_to_plot = "grad_l1_per_task"
        # what_to_plot = "grad_l2_per_task"
        # what_to_plot = "weight_l1_per_task"
        # what_to_plot = "weight_l2_per_task"
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
            # xs = [i * self.avg_interval  for i in range(len(mean_list))]
            xs = [i * 4 for i in range(self.n_tasks)]
            plt.plot(xs, mean_list, label=learner_name, linewidth=2)
            plt.fill_between(xs, mean_list - std_list, mean_list + std_list, alpha=0.1)
        plt.xlabel(f"Task Number", fontsize=22)
        if self.metric == "accuracy":
            # plt.ylabel("Averaged Online Loss", fontsize=22)
            plt.ylabel("Averaged Online Accuracy", fontsize=22)
            # plt.ylabel("Averaged Plasticity", fontsize=22)
            # plt.ylabel("% of Zero Activations", fontsize=22)
            # plt.ylabel(r"$\ell_0$ Norm of Gradients", fontsize=22)
            # plt.ylabel(r"$\ell_1$ Norm of Gradients", fontsize=22)
            # plt.ylabel(r"$\ell_2$ Norm of Gradients", fontsize=22)
            # plt.ylabel(r"$\ell_1$ Norm of Weights", fontsize=22)
            # plt.ylabel(r"$\ell_2$ Norm of Weights", fontsize=22)
        elif self.metric == "loss":
            plt.ylabel("Loss")
        else:
            raise Exception("metric must be loss or accuracy")
        # plt.legend()
        # for input-permuted mnist
        # plt.ylim(bottom=0.68, top=0.80)
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

    # best_runs1 = BestRun("ex8_label_permuted_cifar10", "area", "convolutional_network_relu", [
    #                                                                                             "upgd_fo_global",
    #                                                                                             "upgd_nonprotecting_fo_global",
    #                                                                                              ]).get_best_run(measure="accuracies")

    best_runs1 = [
    'logs/ex8_label_permuted_cifar10/upgd_fo_global/convolutional_network_relu/lr_0.01_beta_utility_0.999_sigma_0.0_weight_decay_0.0',
    'logs/ex8_label_permuted_cifar10/upgd_fo_global/convolutional_network_relu/lr_0.01_beta_utility_0.999_sigma_0.0_weight_decay_0.0001',
    'logs/ex8_label_permuted_cifar10/upgd_fo_global/convolutional_network_relu/lr_0.01_beta_utility_0.999_sigma_0.01_weight_decay_0.0',
    'logs/ex8_label_permuted_cifar10/upgd_fo_global/convolutional_network_relu_with_hooks/lr_0.01_beta_utility_0.999_sigma_0.001_weight_decay_0.0',

    ]


    print(best_runs1)
    plotter = Plotter(best_runs1, metric="accuracy")
    plotter.plot()