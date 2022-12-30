import torch

class FeatureRandomUtility:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
        self.name = 'random_utility'
        
    def compute_utility(self):
        with torch.no_grad():
            weight_utility_net = []
            for name, p in  self.network.named_parameters():
                if 'gate' in name:
                    weight_utility = torch.rand_like(p.data)
                    weight_utility_net.append(torch.mean(weight_utility, dim=0))
            return weight_utility_net  