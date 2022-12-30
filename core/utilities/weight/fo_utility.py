import torch

class FirstOrderUtility:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
        self.name = 'fo_utility'
        
    def compute_utility(self):
        with torch.no_grad():
            fo_utility_net = []
            for p in  self.network.parameters():
                fo_utility = - p.data * p.grad                
                fo_utility_net.append(torch.mean(fo_utility, dim=0))
            return fo_utility_net  
