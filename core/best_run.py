import os
import json
import numpy as np
import re
class BestRun:
    def __init__(self, task_name, metric, network_name, learners):
        self.task_name = task_name
        self.metric = metric
        self.network_name = network_name
        self.learners = learners
    
    def get_best_run(self, measure="losses"):
        best_configs = []
        for learner in self.learners:
            path = f"logs/{self.task_name}/{learner}/{self.network_name}/"
            subdirectories = [f.path for f in os.scandir(path) if f.is_dir()]

            configs = {}
            # loop over all hyperparameters configurations
            for subdirectory in subdirectories:
                # if 'beta_weight_0.9999' in subdirectory:
                #     continue
                # pattern = r'(\w+)_([\d.]+)'
                # matches = re.findall(pattern, subdirectory)
                # result_dict = {key: float(value) for key, value in matches}
                # # if not result_dict["_weight_decay"] == 0.0:
                # if not result_dict["_sigma"] == 0.0:
                if "beta_weight_0.9999" in subdirectory:
                    continue
                configs[subdirectory] = {}
                seeds = os.listdir(f'{subdirectory}')
                configuration_list = []
                for seed in seeds:
                    with open(f"{subdirectory}/{seed}") as json_file:
                        data = json.load(json_file)
                        configuration_list.append(data[measure])

                mean_list = np.nan_to_num(np.array(configuration_list), nan=np.iinfo(np.int32).max).mean(axis=-1)
                configs[subdirectory]["means"] = mean_list
            if measure == "losses":
                best_configs.append(min(configs.keys(), key=(lambda k: sum(configs[k]["means"]))))
            elif measure == "accuracies":
                best_configs.append(max(configs.keys(), key=(lambda k: sum(configs[k]["means"]))))
            else:
                raise Exception("measure must be loss or accuracy")
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
    best_configs = BestRun(args.task_name, args.metric, args.network_name, args.learners).get_best_run(measure="losses")
    print(best_configs)
