import torch

class FeatureOracleUtility:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
        self.name = 'oracle_utility'
        
    def compute_utility(self, original_loss, inputs, targets):
        with torch.no_grad():
            true_utility_net = []
            for name, p in  self.network.named_parameters():
                if 'gate' in name:
                    true_utility = torch.zeros_like(p.data)
                    for i, value in enumerate(p.ravel()):
                        old_value = value.clone()
                        p.ravel()[i] = 0.0
                        output = self.network(inputs)
                        loss = self.criterion(output, targets)
                        p.ravel()[i] = old_value
                        true_utility.ravel()[i] = loss - original_loss
                    true_utility_net.append(torch.mean(true_utility, dim=0))
            return true_utility_net
