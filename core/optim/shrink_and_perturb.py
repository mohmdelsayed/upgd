import torch

class ShrinkandPerturb(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, decay=0.01, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, sigma=sigma, decay=decay, names=names)
        super(ShrinkandPerturb, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                p.data.mul_(1 - group["lr"] * group["decay"]).add_(p.grad + torch.randn_like(p.grad) * group["sigma"], alpha=-group["lr"])
