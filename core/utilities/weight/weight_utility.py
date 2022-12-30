import torch

class WeightUtility:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
        self.name = 'weight_utility'
        
    def compute_utility(self):
        with torch.no_grad():
            weight_utility_net = []
            for p in  self.network.parameters():
                weight_utility = torch.abs(p.data)
                weight_utility_net.append(torch.mean(weight_utility, dim=0))
            return weight_utility_net  
