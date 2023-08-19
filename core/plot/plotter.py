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
    def __init__(self, best_runs_path, metric, task_length=200, n_aggregate_tasks=1):
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
                    elif self.metric == "loss":
                        # configuration_list.append(data["losses"])
                        # configuration_list.append(data["held_out_averages"])
                        configuration_list.append(data["held_out_averages1"])
                        # configuration_list.append(data["held_out_averages2"])
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

    def plot_n_tasks(self, n_tasks=5, avg_interval=1, first_time=1400//4, second_time=0//4):
        colors = [
                  'tab:red',
                  'tab:blue',
                  'tab:gray',
                  ]
        print(self.best_runs_path)
        for color, subdir in zip(colors, self.best_runs_path):
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    if self.metric == "accuracy":
                        configuration_list.append(data["accuracies"])
                    elif self.metric == "loss":
                        # configuration_list.append(data["losses"])
                        # configuration_list.append(data["held_out_averages"])
                        configuration_list.append(data["held_out_averages1"])
                        # configuration_list.append(data["held_out_averages2"])
                    else:
                        raise Exception("metric must be loss or accuracy")
                    learner_name = data["learner"]

            all_data_binned = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // avg_interval, avg_interval).mean(axis=-1)
            all_data_mean = all_data_binned.mean(axis=0)
            all_data_std = all_data_binned.std(axis=0)

            # mean_list1 = all_data_mean[first_time : first_time + (n_tasks * self.task_length)//avg_interval ]
            # std_list1 = all_data_std[first_time : first_time + (n_tasks * self.task_length) // avg_interval] / np.sqrt(len(seeds) * avg_interval)
            mean_list2 = all_data_mean[second_time : second_time + (n_tasks * self.task_length) // avg_interval]
            std_list2 = all_data_std[second_time : second_time + (n_tasks * self.task_length) // avg_interval] / np.sqrt(len(seeds) * avg_interval)

            if 'upgd' in subdir:
                plt.plot(mean_list2, label=learner_name, color=color, alpha=1.0, linewidth=0.5)
            elif 'tab:gray' in color:
                plt.plot(mean_list2, label=learner_name, color=color, alpha=1.0, linewidth=1.0)
            else:
                plt.plot(mean_list2, label=learner_name, color=color, alpha=0.9, linewidth=2.0)
            plt.fill_between([i for i in range(len(mean_list2))], mean_list2 - std_list2, mean_list2 + std_list2, alpha=0.3, color=color)

            # concat_mean = np.concatenate([mean_list1, mean_list2])
            # concat_std = np.concatenate([std_list1, std_list2])
            # plt.plot(concat_mean, label=learner_name, color=color, alpha=0.9)
            # plt.fill_between([i for i in range(len(concat_mean))], concat_mean - concat_std, concat_mean + concat_std, alpha=0.3, color=color)

        plt.xlabel(f"Task Number", fontsize=20)
        if self.metric == "accuracy":
            plt.ylabel("Accuracy", fontsize=20)
        elif self.metric == "loss":
            plt.ylabel("Loss", fontsize=20)
        else:
            raise Exception("metric must be loss or accuracy")
        plt.ylim(bottom=0.0)
        plt.savefig("individual_tasks.pdf", bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":
    # best_runs1 = [
    #               'logs/ex3_toy_permuted_inputset/sgd/fully_connected_linear/lr_0.01',
    #               'logs/ex3_toy_permuted_inputset/sgd/linear_layer/lr_0.01',
    #               'logs/ex3_toy_permuted_inputset_restart/sgd/fully_connected_linear/lr_0.01',
    # ]

    # best_runs1 = [
    #         #   'logs/ex4_changing_average/sgd/linear_layer/lr_0.01',
    #             'logs/ex4_changing_average/upgd_fo_global/fully_connected_linear/lr_0.01_beta_utility_0.9_sigma_0.1',
    #             'logs/ex4_changing_average_restart/sgd/fully_connected_linear/lr_0.01',
    #             'logs/ex4_changing_average/sgd/fully_connected_linear/lr_0.01',
    # ]ex6_input_permuted_mnist
    # ex7_label_permuted_mnist
    # best_runs1 = BestRun("ex6_input_permuted_mnist", "area", "fully_connected_relu", ["adaptive_upgd"]).get_best_run(measure="accuracies")
    # best_runs1 = BestRun("ex5_stationary_mnist", "area", "fully_connected_relu", ["search_fo_anti_corr_global", "search_fo_anti_corr_global2", "search_fo_anti_corr_global3"]).get_best_run(measure="accuracies")

    best_runs1 = BestRun("ex4_toy_changing_outputs_offline", "area", "two_headed_network", ["sgd"]).get_best_run(measure="losses")

    # ex11_split_cub
    # ex10_split_stanford_cars
    # best_runs1 = BestRun("ex4_toy_changing_outputs", "area", "fully_connected_linear", ["noisy_online_ewc_plus", "online_ewc_plus", "si", "noisy_si", "online_ewc", "noisy_online_ewc", "mas", "noisy_mas"]).get_best_run(measure="losses")
    # best_runs1 = BestRun("ex10_split_stanford_cars", "area", "fully_connected_relu", ["sgd", "upgd_fo_global", "online_ewc"]).get_best_run(measure="accuracies")
    # best_runs1 = BestRun("ex10_split_stanford_cars", "area", "fully_connected_relu", ["sgd"]).get_best_run(measure="accuracies")

    # best_runs2 = BestRun("ex4_changing_average_restart", "area", "fully_connected_linear", ["sgd"]).get_best_run(measure="losses")

    # best_runs1 = BestRun("ex6_input_permuted_mnist", "area", "fully_connected_relu", ["adam_upgd2", "adam2", "adam_upgd3"]).get_best_run(measure="accuracies")
    # best_runs1 = BestRun("ex7_label_permuted_mnist", "area", "fully_connected_relu", ["adam2", "idbd", "idbd2"]).get_best_run(measure="accuracies")


    # best_runs1 = BestRun("ex3_toy_permuted_inputset", "area", "fully_connected_linear", ["sgd", "online_ewc"]).get_best_run(measure="losses")
    # best_runs1 = [
    #                 'logs/ex3_toy_permuted_inputset/sgd/fully_connected_linear/lr_0.01',
    #                 'logs/ex3_toy_permuted_inputset/online_ewc/fully_connected_linear/lr_0.01_lamda_1.0_beta_weight_0.999_beta_fisher_0.99',
    #                 'logs/ex3_toy_permuted_inputset/upgd_fo_global/fully_connected_linear/lr_0.01_beta_utility_0.9_sigma_0.0001',
    #                 'logs/ex3_toy_permuted_inputset/upgd_so_global/fully_connected_linear/lr_0.01_beta_utility_0.0_sigma_0.1',
    #                 ]
    # best_runs1 = [
    #             'logs/ex4_changing_average/sgd/fully_connected_linear/lr_0.01',
    #             'logs/ex4_changing_average/online_ewc/fully_connected_linear/lr_0.01_lamda_0.1_beta_weight_0.9999_beta_fisher_0.999',
    #             'logs/ex4_changing_average/upgd_fo_global/fully_connected_linear/lr_0.01_beta_utility_0.9_sigma_0.1',
    #             'logs/ex4_changing_average/upgd_so_global/fully_connected_linear/lr_0.01_beta_utility_0.0_sigma_0.1',
    # ]
    # best_runs1 = BestRun("ex4_changing_average", "area", "fully_connected_linear", ["online_ewc"]).get_best_run(measure="losses")


    print(best_runs1)
    # plotter = Plotter(best_runs1, metric="accuracy")
    plotter = Plotter(best_runs1, metric="loss")
    plotter.plot()
    plotter.plot_n_tasks()