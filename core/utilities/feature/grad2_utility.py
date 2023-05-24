import torch

class FeatureSquaredGradUtility:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
        self.name = 'g2_utility'
        
    def compute_utility(self):
        with torch.no_grad():
            g2_utility_net = []
            for name, p in  self.network.named_parameters():
                if 'gate' in name:
                    g2_utility = p.grad ** 2
                    g2_utility_net.append(g2_utility)
            return g2_utility_net  