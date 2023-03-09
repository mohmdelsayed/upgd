import torch.nn as nn
from .gate import GateLayer
import torch, math

class FullyConnectedReLUGates(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=300):
        super(FullyConnectedReLUGates, self).__init__()
        self.name = "fully_connected_relu_gates"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.ReLU())
        self.add_module("gate_1", GateLayer(n_hidden_units))
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units // 2))
        self.add_module("act_2", nn.ReLU())
        self.add_module("gate_2", GateLayer(n_hidden_units // 2))
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units // 2, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name
    
    def zero_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.zeros_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def const_init(self, const=0.1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight, const)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, const)

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
                m.reset_parameters()

    def __str__(self):
        return self.name
    
    def zero_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.zeros_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def const_init(self, const=0.1):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.constant_(m.weight, const)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, const)

class SmallFullyConnectedReLUGates(nn.Sequential):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=50):
        super(SmallFullyConnectedReLUGates, self).__init__()
        self.name = "small_fully_connected_relu_gates"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.ReLU())
        self.add_module("gate_1", GateLayer(n_hidden_units))
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units))
        self.add_module("act_2", nn.ReLU())
        self.add_module("gate_2", GateLayer(n_hidden_units))
        self.add_module("linear_3", nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name

class SmallFullyConnectedReLU(nn.Sequential):
    def __init__(self, n_obs=4, n_outputs=1, n_hidden_units=50):
        super(SmallFullyConnectedReLU, self).__init__()
        self.name = "small_fully_connected_relu"
        self.add_module("linear_1", nn.Linear(in_features=n_obs, out_features=n_hidden_units))
        self.add_module("act_1", nn.ReLU())
        self.add_module("linear_2", nn.Linear(in_features=n_hidden_units, out_features=n_outputs))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def __str__(self):
        return self.name

if __name__ == "__main__":
    net = FullyConnectedReLU()
    print(net)
