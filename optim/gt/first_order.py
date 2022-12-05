import torch
from torch.nn import functional as F

class ExtendedSGD(torch.optim.SGD):
    def __init__(self, params, lr=1e-5):
        super(ExtendedSGD, self).__init__(params, lr=lr)
    def step(self, loss):
        super().step()

class ExtendedAntiPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5):
        defaults = dict(lr=lr)
        super(ExtendedAntiPGD, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["prev_noise"] = torch.zeros_like(p.data)

                new_noise = torch.randn_like(p.grad)
                perturbed_gradient = p.grad + (new_noise-state["prev_noise"])
                state["prev_noise"] = new_noise
                p.data.add_(perturbed_gradient, alpha=-group["lr"])
                
# UPGD: Utilited-based Perturbed Gradient Descent: variation 1 (utility doesn't control gradient)
class FirstOrderUPGDv1Normal(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0):
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma)
        super(FirstOrderUPGDv1Normal, self).__init__(params, defaults)
    def step(self, loss):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["max_utility"] = torch.tensor(-torch.inf)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                avg_utility = state["avg_utility"]
                noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                avg_utility.mul_(group["beta_utility"]).add_(-p.grad.data * p.data, alpha=1 - group["beta_utility"])
                current_max = avg_utility.max()
                if state["max_utility"] < current_max:
                    state["max_utility"] = current_max
                scaled_utility = torch.tanh_((avg_utility / bias_correction) / group["temp"])
                p.data.add_(p.grad.data + noise * (1 - scaled_utility/ torch.tanh(state["max_utility"])), alpha=-group["lr"])

# UPGD: Utilited-based Perturbed Gradient Descent: variation 1 with Anti-correlated noise 
class FirstOrderUPGDv1AntiCorr(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0):
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma)
        super(FirstOrderUPGDv1AntiCorr, self).__init__(params, defaults)
    def step(self, loss):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["prev_noise"] = torch.zeros_like(p.data)
                    state["max_utility"] = torch.tensor(-torch.inf)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                noise = (new_noise-state["prev_noise"])
                state["prev_noise"] = new_noise
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(-p.grad.data * p.data, alpha=1 - group["beta_utility"])
                current_max = avg_utility.max()
                if state["max_utility"] < current_max:
                    state["max_utility"] = current_max
                # scaled_utility = torch.tanh_((avg_utility / bias_correction) / group["temp"]) / torch.tanh(state["max_utility"])
                scaled_utility = torch.tanh_((avg_utility / bias_correction) / group["temp"] / state["max_utility"])
                p.data.add_(p.grad.data + noise * (1 - scaled_utility), alpha=-group["lr"])

# UPGD: Utilited-based Perturbed Gradient Descent: variation 1 with Anti-correlated noise 
class FirstOrderUPGDv1AntiCorrNormalized(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0):
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma)
        super(FirstOrderUPGDv1AntiCorrNormalized, self).__init__(params, defaults)
    def step(self, loss):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["prev_noise"] = torch.zeros_like(p.data)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                noise = (new_noise-state["prev_noise"])
                state["prev_noise"] = new_noise
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(-p.grad.data * p.data, alpha=1 - group["beta_utility"])
                scaled_utility = torch.tanh_(F.normalize((avg_utility / bias_correction), dim=-1) / group["temp"])
                p.data.add_(p.grad.data + noise * (1 - scaled_utility), alpha=-group["lr"])

# UPGD: Utilited-based Perturbed Gradient Descent: variation 2 (utility controls gradient)
class FirstOrderUPGDv2Normal(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0):
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma)
        super(FirstOrderUPGDv2Normal, self).__init__(params, defaults)
    def step(self, loss):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["max_utility"] = torch.tensor(-torch.inf)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                avg_utility = state["avg_utility"]
                noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                avg_utility.mul_(group["beta_utility"]).add_(-p.grad.data * p.data, alpha=1 - group["beta_utility"])
                current_max = avg_utility.max()
                if state["max_utility"] < current_max:
                    state["max_utility"] = current_max
                scaled_utility = torch.tanh_((avg_utility / bias_correction) / group["temp"])
                p.data.add_((p.grad.data + noise) * (1 - scaled_utility/ torch.tanh(state["max_utility"])), alpha=-group["lr"])

# UPGD: Utilited-based Perturbed Gradient Descent: variation 2 with Anti-correlated noise 
class FirstOrderUPGDv2AntiCorr(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0):
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma)
        super(FirstOrderUPGDv2AntiCorr, self).__init__(params, defaults)
    def step(self, loss):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["prev_noise"] = torch.zeros_like(p.data)
                    state["max_utility"] = torch.tensor(-torch.inf)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                noise = (new_noise-state["prev_noise"])
                state["prev_noise"] = new_noise
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(-p.grad.data * p.data, alpha=1 - group["beta_utility"])
                current_max = avg_utility.max()
                if state["max_utility"] < current_max:
                    state["max_utility"] = current_max
                scaled_utility = torch.tanh_((avg_utility / bias_correction) / group["temp"] / state["max_utility"])
                p.data.add_((p.grad.data + noise) * (1 - scaled_utility), alpha=-group["lr"])


# UPGD: Utilited-based Perturbed Gradient Descent: variation 2 with Anti-correlated noise 
class FirstOrderUPGDv2AntiCorrNormalized(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0):
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma)
        super(FirstOrderUPGDv2AntiCorrNormalized, self).__init__(params, defaults)
    def step(self, loss):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["prev_noise"] = torch.zeros_like(p.data)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                noise = (new_noise-state["prev_noise"])
                state["prev_noise"] = new_noise
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(-p.grad.data * p.data, alpha=1 - group["beta_utility"])
                scaled_utility = torch.tanh_(F.normalize((avg_utility / bias_correction), dim=-1) / group["temp"])
                p.data.add_((p.grad.data + noise) * (1 - scaled_utility), alpha=-group["lr"])


# UGD: Utility-regularized Gradient Descent
class FirstOrderUGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, lamda=1.0, beta_utility=0.9, beta_weight=0.9):
        defaults = dict(lr=lr, lamda=lamda, beta_utility=beta_utility, beta_weight=beta_weight)
        super(FirstOrderUGD, self).__init__(params, defaults)
    def step(self, loss):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["avg_weight"] = torch.zeros_like(p.data)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                avg_utility = state["avg_utility"]
                avg_weight = state["avg_weight"]
                utility = - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(utility, alpha=1 - group["beta_utility"])
                avg_weight.mul_(group["beta_weight"]).add_(p.data, alpha=1 - group["beta_weight"])
                p.data.add_(p.grad.data + group["lamda"] * avg_utility * (p.data - avg_weight), alpha=-group["lr"] / bias_correction)
