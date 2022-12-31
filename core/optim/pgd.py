import torch


class ExtendedPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, sigma=1.0):
        _, params = zip(*params)
        defaults = dict(lr=lr, sigma=sigma)
        super(ExtendedPGD, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for p in group["params"]:
                perturbed_gradient = p.grad + torch.randn_like(p.grad) * group["sigma"]
                p.data.add_(perturbed_gradient, alpha=-group["lr"])
