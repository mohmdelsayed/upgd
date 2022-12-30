import torch

class OracleUtility:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
        self.name = 'oracle_utility'
        
    def compute_utility(self, original_loss, inputs, targets):
        with torch.no_grad():
            true_utility_net = []
            for p in  self.network.parameters():
                true_utility = torch.zeros_like(p.data)
                for i, value in enumerate(p.ravel()):
                    old_value = value.clone()
                    p.ravel()[i] = 0.0
                    output = self.network(inputs)
                    loss = self.criterion(output, targets)
                    p.ravel()[i] = old_value
                    true_utility.ravel()[i] = loss - original_loss
                # true_utility = torch.argsort(true_utility.ravel(), dim=-1).reshape(p.data.shape)
                true_utility_net.append(true_utility)
            return true_utility_net
