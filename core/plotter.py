import json
import matplotlib.pyplot as plt
from core.best_run import BestRun
import os
import numpy as np

class Plotter:
    def __init__(self, best_runs_path):
        self.best_runs_path = best_runs_path

    def plot(self):
        for subdir in self.best_runs_path:
            seeds = os.listdir(f'{subdir}')
            configuration_list = []
            for seed in seeds:
                with open(f"{subdir}/{seed}") as json_file:
                    data = json.load(json_file)
                    configuration_list.append(data["losses"])
                    learner_name = data["learner"]

            mean_list = np.array(configuration_list).mean(axis=-1)
            std_list = np.array(configuration_list).std(axis=-1) / np.sqrt(len(seeds))
            plt.plot(mean_list, label=learner_name)
            plt.fill_between(range(len(mean_list)), mean_list - std_list, mean_list + std_list, alpha=0.1)
            plt.legend()
        
        plt.xlabel("Time Step")
        plt.ylabel("Loss")
        plt.show()

if __name__ == "__main__":
    best_runs = BestRun("summer_with_sign_change", "area", "fully_connected_tanh", ["sgd", "upgdv2_normalized"]).get_best_run()
    print(best_runs)
    plotter = Plotter(best_runs)
    plotter.plot()