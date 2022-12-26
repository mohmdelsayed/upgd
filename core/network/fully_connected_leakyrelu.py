import torch
import torch.nn as nn


class FullyConnectedLeakyReLU(nn.Module):
    def __init__(self, n_obs=10, n_outputs=10, n_hidden_units=300):
        super(FullyConnectedLeakyReLU, self).__init__()
        self.name = "fully_connected_leakyrelu"
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_obs, n_hidden_units, True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(n_hidden_units, n_hidden_units // 2, True),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(n_hidden_units // 2, n_outputs, True),
        )

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return self.name

if __name__ == "__main__":
    net = FullyConnectedLeakyReLU()
    print(net)
