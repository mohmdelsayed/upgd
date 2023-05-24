import torch
from torch.nn import functional as F

class RandomSearchUncorrelated(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, sigma=sigma, names=names)
        super(RandomSearchUncorrelated, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                noise = torch.randn_like(p.grad) * group["sigma"]
                p.data.add_(noise, alpha=-group["lr"])


class RandomSearchAnticorrelated(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, sigma=sigma, names=names)
        super(RandomSearchAnticorrelated, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["prev_noise"] = torch.zeros_like(p.data)
                new_noise = torch.randn_like(p.grad) * group["sigma"]
                noise = new_noise - state["prev_noise"]
                state["prev_noise"] = new_noise
                p.data.add_(noise, alpha=-group["lr"])