import json
import os


class Logger:
    """
    This class is responsible for logging the training process.
    Log file will be saved in the log directory with JSON format.
    """

    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir

    def log(self, **kwargs):
        json_object = json.dumps(kwargs, indent=4)
        file_name = ''
        for key, value in kwargs["optimizer_hps"].items():
            file_name += f'{key}_{value}_'
        file_name = file_name[:-1]
        
        dir = f"{self.log_dir}/{kwargs['task']}/{kwargs['learner']}/{kwargs['network']}/{file_name}/"
        if not os.path.exists(dir):
            os.makedirs(dir)

        with open(f"{dir}/{kwargs['seed']}.json", "w") as outfile:
            outfile.write(json_object)

