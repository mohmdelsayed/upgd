import torch.nn as nn
from .gate import GateLayer
import torch, math

class FullyConnectedLinearGates(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=100):
        super(FullyConnectedLinearGates, self).__init__()
        self.name = "fully_connected_linear_gates"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("gate_1", GateLayer(n_hidden_units))
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units // 2))
        self.add_module("gate_2", GateLayer(n_hidden_units // 2))
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units // 2, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
class FullyConnectedLinear(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=100):
        super(FullyConnectedLinear, self).__init__()
        self.name = "fully_connected_linear"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units // 2))
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units // 2, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name

if __name__ == "__main__":
    net = FullyConnectedLinear()
    print(net)
