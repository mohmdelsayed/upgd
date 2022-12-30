import torch
import torch.nn as nn

class GateLayer(nn.Module):
    def __init__(self, input_features):
        super(GateLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_features), requires_grad=True)
    def forward(self, input):
        return input * self.weight