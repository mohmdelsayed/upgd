import json
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, log_path):
        self.log_path = log_path

    def plot(self):
        with open(self.log_path) as json_file:
            data = json.load(json_file)
            print(data)
            plt.plot(data["train_loss"], label="train_loss")
            plt.plot(data["val_loss"], label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()