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

    def plot(self):
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            utilities = {}
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    configuration_list.append(data["losses"])
                    learner_name = data["learner"]
                    for measure in data["utilities"]:
                        if measure not in utilities:
                            utilities[measure] = []
                        utilities[measure].append(data["utilities"][measure])

            configuration_list = np.array(configuration_list).reshape(len(seeds), len(configuration_list[0]) // self.avg_interval, self.avg_interval).mean(axis=-1)

            mean_list = np.array(configuration_list).mean(axis=0)
            std_list = np.array(configuration_list).std(axis=0) / np.sqrt(len(seeds))


            plt.plot(mean_list, label=learner_name)
            plt.fill_between(range(len(mean_list)), mean_list - std_list, mean_list + std_list, alpha=0.1)
            plt.legend()

            for measure in utilities:
                utilities[measure] = np.array(utilities[measure]).reshape(len(seeds), len(utilities[measure][0]) // self.avg_interval, self.avg_interval).mean(axis=-1)

                mean_measure = np.array(utilities[measure]).mean(axis=0)
                std_measure = np.array(utilities[measure]).std(axis=0) / np.sqrt(len(seeds))
                plt.plot(mean_measure, label=f"{measure}")
                plt.fill_between(range(len(mean_measure)), mean_measure - std_measure, mean_measure + std_measure, alpha=0.1)
                plt.legend()

        
        plt.xlabel(f"Bin ({self.avg_interval} samples each)")
        plt.ylabel("Loss")
        plt.show()

if __name__ == "__main__":
    best_runs = BestRun("utility_task_batch=1", "area", "small_fully_connected_tanh", ["sgd"]).get_best_run()
    print(best_runs)
    plotter = UtilityPlotter(best_runs)
    plotter.plot()