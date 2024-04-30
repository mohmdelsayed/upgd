import torch

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, names=names)
        super(SGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                p.data.add_(p.grad + group['weight_decay'] * p.data, alpha=-group["lr"])