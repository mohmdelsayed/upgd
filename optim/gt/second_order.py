from hesscale import HesScale
import torch

# UPGD: Utilited-based Perturbed Gradient Descent: variation 1 (utility doesn't control gradient)
class SecondOrderUPGDv1Normal(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.999, temp=5.0):
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, method_field=type(self).method.savefield)
        super(SecondOrderUPGDv1Normal, self).__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                avg_utility = state["avg_utility"]
                noise = torch.randn_like(p.grad)
                hess_param = getattr(p, group["method_field"])
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(utility, alpha=1 - group["beta_utility"])
                p.data.add_(p.grad.data + noise * (1-torch.tanh_(avg_utility / group["temp"])), alpha=-group["lr"] / bias_correction)

# UPGD: Utilited-based Perturbed Gradient Descent: variation 1 with Anti-correlated noise 
class SecondOrderUPGD1AntiCorr(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.999, temp=5.0):
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, method_field=type(self).method.savefield)
        super(SecondOrderUPGD1AntiCorr, self).__init__(params, defaults)
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
                hess_param = getattr(p, group["method_field"])
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(utility, alpha=1 - group["beta_utility"])
                p.data.add_(p.grad.data + noise * (1-torch.tanh_(avg_utility / group["temp"])), alpha=-group["lr"] / bias_correction)

# UPGD: Utilited-based Perturbed Gradient Descent: variation 2 (utility controls gradient)
class SecondOrderUPGDv2Normal(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.999, temp=1.0):
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, method_field=type(self).method.savefield)
        super(SecondOrderUPGDv2Normal, self).__init__(params, defaults)
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
                hess_param = getattr(p, group["method_field"])
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(utility, alpha=1 - group["beta_utility"])
                p.data.add_((p.grad.data + noise) * (1-torch.tanh_(avg_utility / group["temp"])), alpha=-group["lr"] / bias_correction)

# UPGD: Utilited-based Perturbed Gradient Descent: variation 2 with Anti-correlated noise 
class SecondOrderUPGD2AntiCorr(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.999, temp=5.0):
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, method_field=type(self).method.savefield)
        super(SecondOrderUPGD2AntiCorr, self).__init__(params, defaults)
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
                hess_param = getattr(p, group["method_field"])
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(utility, alpha=1 - group["beta_utility"])
                p.data.add_(p.grad.data + noise * (1-torch.tanh_(avg_utility / group["temp"])), alpha=-group["lr"] / bias_correction)

# UGD: Utility-regularized Gradient Descent
class SecondOrderUGD(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, lamda=1.0, beta_utility=0.999, beta_weight=0.999):
        defaults = dict(lr=lr, lamda=lamda, beta_utility=beta_utility, beta_weight=beta_weight, method_field=type(self).method.savefield)
        super(SecondOrderUGD, self).__init__(params, defaults)
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["avg_weight"] = torch.zeros_like(p.data)
                avg_utility = state["avg_utility"]
                avg_weight = state["avg_weight"]
                hess_param = getattr(p, group["method_field"])
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(utility, alpha=1 - group["beta_utility"])
                avg_weight.mul_(group["beta_weight"]).add_(p.data, alpha=1 - group["beta_weight"])
                p.data.add_(p.grad.data + group["lamda"] * avg_utility * (p.data - avg_weight), alpha=-group["lr"])
