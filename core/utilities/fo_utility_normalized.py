import torch
from torch.nn import functional as F

class FirstOrderUtilityNormalized:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
        self.name = 'fo_utility_normalized'
        
    def compute_utility(self):
        with torch.no_grad():
            fo_utility_net = []
            for p in  self.network.parameters():
                fo_utility = - p.data * p.grad   
                # fo_utility = torch.argsort(fo_utility.ravel(), dim=-1).reshape(p.data.shape)
                fo_utility_net.append(F.normalize(fo_utility, dim=-1))
            return fo_utility_net  
