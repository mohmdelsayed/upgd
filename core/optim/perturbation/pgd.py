import torch


class PGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5):
        defaults = dict(lr=lr)
        super(PGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                perturbed_gradient = p.grad + torch.randn_like(p.grad)
                p.data.add_(perturbed_gradient, alpha=-group["lr"])
