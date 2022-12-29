import torch.nn as nn

class FullyConnectedReLU(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=300):
        super(FullyConnectedReLU, self).__init__()
        self.name = "fully_connected_relu"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.ReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units // 2))
        self.add_module("act_2", nn.ReLU())
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units // 2, out_features=n_outputs))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def __str__(self):
        return self.name

class SmallFullyConnectedReLU(nn.Sequential):
    def __init__(self, n_obs=4, n_outputs=1, n_hidden_units=10):
        super(SmallFullyConnectedReLU, self).__init__()
        self.name = "small_fully_connected_relu"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.ReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

    def __str__(self):
        return self.name

if __name__ == "__main__":
    net = FullyConnectedReLU()
    print(net)
