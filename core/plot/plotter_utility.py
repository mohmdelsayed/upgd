import json
import matplotlib.pyplot as plt
from core.best_run import BestRun
import os
import numpy as np
from core.plotter import Plotter

class UtilityPlotter(Plotter):
    def __init__(self, best_runs_path, avg_interval=50):
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


            plt.plot(mean_list, label=learner_name)
            plt.fill_between(range(len(mean_list)), mean_list - std_list, mean_list + std_list, alpha=0.1)
            plt.legend()

            for measure in util_correlations:
                util_correlations[measure] = np.array(util_correlations[measure]).reshape(len(seeds), len(util_correlations[measure][0]) // self.avg_interval, self.avg_interval).mean(axis=-1)

                mean_measure = np.array(util_correlations[measure]).mean(axis=0)
                std_measure = np.array(util_correlations[measure]).std(axis=0) / np.sqrt(len(seeds))
                plt.plot(mean_measure, label=f"{measure}")
                plt.fill_between(range(len(mean_measure)), mean_measure - std_measure, mean_measure + std_measure, alpha=0.1)
                plt.title(corr_name)
                plt.legend()

        
        plt.xlabel(f"Bin ({self.avg_interval} samples each)")
        plt.ylabel("Loss")
        plt.show()

if __name__ == "__main__":
    best_runs = BestRun("utility_task", "area", "fully_connected_tanh_gates", ["sgd"]).get_best_run()
    print(best_runs)
    plotter = UtilityPlotter(best_runs)
    plotter.plot("global_correlations")
    plotter.plot("layerwise_correlations")