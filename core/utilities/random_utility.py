import torch

class RandomUtility:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
        self.name = 'random_utility'
        
    def compute_utility(self):
        with torch.no_grad():
            weight_utility_net = []
            for p in  self.network.parameters():
                weight_utility = torch.rand_like(p.data)
                # weight_utility = torch.argsort(weight_utility.ravel(), dim=-1).reshape(p.data.shape)
                weight_utility_net.append(weight_utility)
            return weight_utility_net  