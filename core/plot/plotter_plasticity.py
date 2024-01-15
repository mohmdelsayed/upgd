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
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    if self.metric == "accuracy":
                        configuration_list.append(data["plasticity_per_task"])
                    elif self.metric == "loss":
                        configuration_list.append(data["losses"])
                    else:
                        raise Exception("metric must be loss or accuracy")
                    learner_name = data["learner"]

            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)
            plasticity_list = np.array(configuration_list).mean(axis=0)
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
            accuracy_list = np.array(configuration_list).mean(axis=0)
            
            plasticity_list50 = plasticity_list[-100:]
            plasticity_metric = plasticity_list50[0] - plasticity_list50[-1]
            forgetting_metric = accuracy_list[0] - accuracy_list[-1]
            print(learner_name, plasticity_metric, forgetting_metric)

            plasticity_list = plasticity_list[-50:]
            accuracy_list = accuracy_list[-50:]

            xs = [i * self.avg_interval  for i in range(len(accuracy_list))]
            
            plt.scatter(accuracy_list, plasticity_list, label=learner_name, alpha=0.2)
            b, a = np.polyfit(accuracy_list, plasticity_list, deg=1)
            yfit = [a + b * xi for xi in accuracy_list]
            plt.plot(accuracy_list, yfit, lw=2.5, alpha=1)

        plt.ylabel("Average Online Plasticity", fontsize=24)
        plt.xlabel("Average Online Accuracy", fontsize=24)

        plt.savefig("ss.pdf", bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":
    best_runs1 = BestRun("ex8_label_permuted_cifar10", "area", "convolutional_network_relu_with_hooks", [
                                                                                                 "upgd_fo_global",
                                                                                                 "sgd", 
                                                                                                 "pgd",
                                                                                                 "adam",
                                                                                                 "upgd_nonprotecting_fo_global", 
                                                                                                 "shrink_and_perturb",
                                                                                                 "online_ewc",
                                                                                                 "mas",
                                                                                                 "si_new",
                                                                                                 "rwalk",
                                                                                                 ]).get_best_run(measure="accuracies")

    print(best_runs1)
    plotter = Plotter(best_runs1, metric="accuracy")
    plotter.plot()