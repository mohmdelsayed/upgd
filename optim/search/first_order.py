import torch

# Utility-based Search Optimizers

class FirstOrderSearchNormal(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.999, temp=1.0):
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp)
        super(FirstOrderSearchNormal, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad)
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(-p.grad.data * p.data, alpha=1 - group["beta_utility"])
                p.data.add_(noise * (1-torch.tanh_(avg_utility / group["temp"])), alpha=-group["lr"] / bias_correction)

class FirstOrderSearchAntiCorr(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.999, temp=1.0):
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp)
        super(FirstOrderSearchAntiCorr, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["prev_noise"] = torch.zeros_like(p.data)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                new_noise = torch.randn_like(p.grad)
                noise = (new_noise-state["prev_noise"])
                state["prev_noise"] = new_noise
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(-p.grad.data * p.data, alpha=1 - group["beta_utility"])
                p.data.add_(noise * (1-torch.tanh_(avg_utility / group["temp"])), alpha=-group["lr"] / bias_correction)