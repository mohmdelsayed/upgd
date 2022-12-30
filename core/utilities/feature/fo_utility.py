import torch

class FeatureFirstOrderUtility:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
        self.name = 'fo_utility'
        
    def compute_utility(self):
        with torch.no_grad():
            fo_utility_net = []
            for name, p in  self.network.named_parameters():
                if 'gate' in name:
                    fo_utility = - p.data * p.grad
                    fo_utility_net.append(torch.mean(fo_utility, dim=0))
            return fo_utility_net  
