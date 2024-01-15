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
    def __init__(self, best_runs_path, metric, avg_interval=1):
        self.best_runs_path = best_runs_path
        self.avg_interval = avg_interval
        self.metric = metric
        self.n_tasks = 400

    def plot(self):
        colors = [ 'tab:blue',
                    'tab:red',
                    'tab:orange'
        ]
        what_to_plot = ["accuracies", "held_out_accuracies"]
        for term in what_to_plot:
            for color, subdir in zip(colors, self.best_runs_path):
                seeds = os.listdir(f'{subdir}')
                configuration_list = []
                for seed in seeds:
                    with open(f"{subdir}/{seed}") as json_file:
                        data = json.load(json_file)
                        if self.metric == "accuracy":
                            if "held_out" in term:
                                configuration_list.append([data[term][0]]+data[term])
                            else:
                                configuration_list.append(data[term])
                        elif self.metric == "loss":
                            configuration_list.append(data["losses"])
                        else:
                            raise Exception("metric must be loss or accuracy")
                        learner_name = data["learner"]

                if "held_out" in term:
                    self.avg_interval = 1
                    style = '-'
                else:
                    self.avg_interval = 2500
                    style = '--'
                number_of_tasks_to_plot = 80
                configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
                mean_list = np.array(configuration_list).mean(axis=0)[:number_of_tasks_to_plot]
                x = [i * 1 for i in range(len(mean_list))]
                std_list = np.array(configuration_list).std(axis=0)[:number_of_tasks_to_plot] / np.sqrt(len(seeds))
                plt.plot(x, mean_list, style, label=learner_name, color=color, linewidth=2)
                plt.fill_between(x, mean_list - std_list, mean_list + std_list, alpha=0.1, color=color)
            plt.xlabel(f"Task Number", fontsize=24)
            if self.metric == "accuracy":
                plt.ylabel("Average Accuracy", fontsize=24)
            elif self.metric == "loss":
                plt.ylabel("Loss", fontsize=24)
            else:
                raise Exception("metric must be loss or accuracy")
            # plt.ylim(bottom=0.0)
            plt.xlim(right=80)
        plt.savefig("offline.pdf", bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":

    best_runs1 = [
                    'logs/ex7_label_permuted_mnist_offline/upgd_fo_global/fully_connected_relu/lr_0.02_beta_utility_0.9_sigma_0.001_weight_decay_0.0',
                    'logs/ex7_label_permuted_mnist_offline/adam/fully_connected_relu/lr_0.0001_weight_decay_0.1_beta1_0.0_beta2_0.9999_damping_1e-08',
                    # 'logs/ex7_label_permuted_mnist_offline/sgd/fully_connected_relu/lr_0.01_weight_decay_0.0001',
    ]


    print(best_runs1)
    plotter = Plotter(best_runs1, metric="accuracy")
    plotter.plot()