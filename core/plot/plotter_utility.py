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
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 14})

class UtilityPlotter:
    def __init__(self, best_runs_path, avg_interval=10):
        self.best_runs_path = best_runs_path
        self.avg_interval = avg_interval

    def plot(self, corr_name="global_correlations"):
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            util_correlations = {}
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    configuration_list.append(data["losses"])
                    learner_name = data["learner"]
                    for measure in data[corr_name]:
                        if measure not in util_correlations:
                            util_correlations[measure] = []
                        util_correlations[measure].append(data[corr_name][measure])
            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)

            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))

            x = [self.avg_interval * i for i in range(len(mean_list))]
            plt.plot(x, mean_list, label=learner_name)
            plt.fill_between(x, mean_list - std_list, mean_list + std_list, alpha=0.1)

            # plt.legend()

            colors = ["tab:orange", "tab:green", "tab:red", "tab:brown"]
            x = [self.avg_interval * i for i in range(len(mean_list))]
            for color, measure in zip(colors, util_correlations):
                util_correlations[measure] = np.array(util_correlations[measure]).reshape(len(seeds), len(util_correlations[measure][0]) // self.avg_interval, self.avg_interval).mean(axis=-1)

                mean_measure = np.array(util_correlations[measure]).mean(axis=0)
                std_measure = np.array(util_correlations[measure]).std(axis=0) / np.sqrt(len(seeds))
                plt.plot(x, mean_measure, label=f"{measure}", linewidth=2, color=color)
                plt.fill_between(x, mean_measure - std_measure, mean_measure + std_measure, alpha=0.1, color=color)
                # plt.legend()

        
        plt.xlabel("Number of Samples", fontsize=20)
        plt.ylabel("Spearman's Coefficient", fontsize=20)
        plt.savefig(f"{corr_name}.pdf", bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":
    best_runs = BestRun("weight_utils", "area", "small_fully_connected_relu", ["sgd_with_hesscale"]).get_best_run()
    print(best_runs)
    plotter = UtilityPlotter(best_runs)
    plotter.plot("global_correlations")
    plotter.plot("layerwise_correlations")