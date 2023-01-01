import os
import json
import numpy as np
class BestRun:
    def __init__(self, task_name, metric, network_name, learners):
        self.task_name = task_name
        self.metric = metric
        self.network_name = network_name
        self.learners = learners
    
    def get_best_run(self):
        best_configs = []
        for learner in self.learners:
            path = f"logs/{self.task_name}/{learner}/{self.network_name}/"
            subdirectories = [f.path for f in os.scandir(path) if f.is_dir()]

            configs = {}
            # loop over all hyperparameters configurations
            for subdirectory in subdirectories:
                configs[subdirectory] = {}
                seeds = os.listdir(f'{subdirectory}')
                configuration_list = []
                for seed in seeds:
                    with open(f"{subdirectory}/{seed}") as json_file:
                        data = json.load(json_file)
                        configuration_list.append(data["losses"])

                mean_list = np.nan_to_num(np.array(configuration_list)).mean(axis=-1)
                configs[subdirectory]["means"] = mean_list

            best_configs.append(min(configs.keys(), key=(lambda k: sum(configs[k]["means"]))))

        return best_configs

if __name__ == "__main__":
    # python core/best_run.py --task_name summer_with_sign_change
    # --network_name fully_connected_tanh--learners sgd upgdv2_normalized --metric "area"
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--network_name", type=str, required=True)
    parser.add_argument('--metric', nargs='+', default=[])
    parser.add_argument('--learners', nargs='+', default=[])
    args = parser.parse_args()
    best_configs = BestRun(args.task_name, args.metric, args.network_name, args.learners).get_best_run()
    print(best_configs)
