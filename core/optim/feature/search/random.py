import torch
from torch.nn import functional as F

class RandomSearchUncorrelated(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, sigma=1.0):
        names, params = zip(*params)
        self.prev_gate = False
        defaults = dict(lr=lr, sigma=sigma, names=names)
        super(RandomSearchUncorrelated, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                if 'gate' in name:
                    self.prev_gate = True
                    continue
                if self.prev_gate:
                    noise = torch.randn_like(p.grad) * group["sigma"]
                    p.data.add_(noise, alpha=-group["lr"])
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])

class RandomSearchAnticorrelated(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, sigma=1.0):
        names, params = zip(*params)
        self.prev_gate = False
        defaults = dict(lr=lr, sigma=sigma, names=names)
        super(RandomSearchAnticorrelated, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if len(state) == 0:
                    state["prev_noise"] = torch.zeros_like(p.data)
                if 'gate' in name:
                    self.prev_gate = True
                    continue
                if self.prev_gate:
                    new_noise = torch.randn_like(p.grad) * group["sigma"]
                    noise = new_noise - state["prev_noise"]
                    state["prev_noise"] = new_noise
                    p.data.add_(noise, alpha=-group["lr"])
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])