from hesscale import HesScale
import torch

class SecondOrderSearchNormal(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.999, beta_weight=0.999, temp=1.0):
        defaults = dict(lr=lr, method_field=type(self).method.savefield, beta_utility=beta_utility, beta_weight=beta_weight, temp=temp)
        super(SecondOrderSearchNormal, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["avg_utility"] = torch.zeros_like(p.data)
                avg_utility = state["avg_utility"]
                hess_param = getattr(p, group["method_field"])
                noise = torch.randn_like(p.grad)                
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(utility, alpha=1 - group["beta_utility"])
                p.data.add_(noise * (1-torch.tanh_(avg_utility / group["temp"])), alpha=-group["lr"])

class SecondOrderSearchAntiCorr(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.999, beta_weight=0.999, temp=1.0):
        defaults = dict(lr=lr, method_field=type(self).method.savefield, beta_utility=beta_utility, beta_weight=beta_weight, temp=temp)
        super(SecondOrderSearchAntiCorr, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["prev_noise"] = torch.zeros_like(p.data)

                avg_utility = state["avg_utility"]
                hess_param = getattr(p, group["method_field"])

                new_noise = torch.randn_like(p.grad)
                noise = (new_noise-state["prev_noise"])
                state["prev_noise"] = new_noise
                
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(utility, alpha=1 - group["beta_utility"])
                p.data.add_(noise * (1-torch.tanh_(avg_utility / group["temp"])), alpha=-group["lr"])
