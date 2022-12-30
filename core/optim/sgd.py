import torch

class ExtendedSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5):
        names, params = zip(*params)
        defaults = dict(lr=lr, names=names)
        super(ExtendedSGD, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                p.data.add_(p.grad, alpha=-group["lr"])