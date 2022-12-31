import sys, os
from torch.nn import functional as F
sys.path.insert(1, os.getcwd())
from HesScale.hesscale import HesScale
import torch

class SecondOrderUPGDv1AntiCorrNormalized(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, method_field=type(self).method.savefield, noise_damping=noise_damping, names=names)
        super(SecondOrderUPGDv1AntiCorrNormalized, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["prev_noise"] = torch.zeros_like(p.data)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                if group["noise_damping"]:
                    new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                else:
                    new_noise = torch.randn_like(p.grad) * group["sigma"]
                noise = new_noise - state["prev_noise"]
                state["prev_noise"] = new_noise
                avg_utility = state["avg_utility"]
                hess_param = getattr(p, group["method_field"])
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(
                    utility, alpha=1 - group["beta_utility"]
                )
                scaled_utility = torch.tanh_(
                    F.normalize((avg_utility / bias_correction), dim=-1) / group["temp"]
                )
                p.data.add_(
                    p.grad.data + noise * (1 - scaled_utility), alpha=-group["lr"]
                )

# UPGD: Utilited-based Perturbed Gradient Descent: variation 1 with Anti-correlated noise
class SecondOrderUPGDv1AntiCorrMax(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, method_field=type(self).method.savefield, noise_damping=noise_damping, names=names)
        super(SecondOrderUPGDv1AntiCorrMax, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["prev_noise"] = torch.zeros_like(p.data)
                    state["max_utility"] = torch.tensor(-torch.inf)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                if group["noise_damping"]:
                    new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                else:
                    new_noise = torch.randn_like(p.grad) * group["sigma"]
                noise = new_noise - state["prev_noise"]
                state["prev_noise"] = new_noise
                avg_utility = state["avg_utility"]
                hess_param = getattr(p, group["method_field"])
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(
                    utility, alpha=1 - group["beta_utility"]
                )
                current_max = avg_utility.max()
                if state["max_utility"] < current_max:
                    state["max_utility"] = current_max
                scaled_utility = torch.tanh_((avg_utility / bias_correction) / group["temp"]) / torch.tanh(state["max_utility"])
                p.data.add_(
                    p.grad.data + noise * (1 - scaled_utility), alpha=-group["lr"]
                )
class SecondOrderUPGDv2AntiCorrNormalized(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, method_field=type(self).method.savefield, noise_damping=noise_damping, names=names)
        super(SecondOrderUPGDv2AntiCorrNormalized, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["prev_noise"] = torch.zeros_like(p.data)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                if group["noise_damping"]:
                    new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                else:
                    new_noise = torch.randn_like(p.grad) * group["sigma"]
                noise = new_noise - state["prev_noise"]
                state["prev_noise"] = new_noise
                avg_utility = state["avg_utility"]
                hess_param = getattr(p, group["method_field"])
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(
                    utility, alpha=1 - group["beta_utility"]
                )
                scaled_utility = torch.tanh_(
                    F.normalize((avg_utility / bias_correction), dim=-1) / group["temp"]
                )
                p.data.add_(
                    (p.grad.data + noise) * (1 - scaled_utility), alpha=-group["lr"]
                )

# UPGD: Utilited-based Perturbed Gradient Descent: variation 2 with Anti-correlated noise
class SecondOrderUPGDv2AntiCorrMax(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, method_field=type(self).method.savefield, noise_damping=noise_damping, names=names)
        super(SecondOrderUPGDv2AntiCorrMax, self).__init__(params, defaults)
    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["prev_noise"] = torch.zeros_like(p.data)
                    state["max_utility"] = torch.tensor(-torch.inf)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                if group["noise_damping"]:
                    new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                else:
                    new_noise = torch.randn_like(p.grad) * group["sigma"]
                noise = new_noise - state["prev_noise"]
                state["prev_noise"] = new_noise
                avg_utility = state["avg_utility"]
                hess_param = getattr(p, group["method_field"])
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(
                    utility, alpha=1 - group["beta_utility"]
                )
                current_max = avg_utility.max()
                if state["max_utility"] < current_max:
                    state["max_utility"] = current_max
                scaled_utility = torch.tanh_((avg_utility / bias_correction) / group["temp"]) / torch.tanh(state["max_utility"])
                p.data.add_(
                    (p.grad.data + noise) * (1 - scaled_utility), alpha=-group["lr"]
                )


class SecondOrderUPGDv1NormalNormalized(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, method_field=type(self).method.savefield, noise_damping=noise_damping, names=names)
        super(SecondOrderUPGDv1NormalNormalized, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                if group["noise_damping"]:
                    noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                else:
                    noise = torch.randn_like(p.grad) * group["sigma"]
                avg_utility = state["avg_utility"]
                hess_param = getattr(p, group["method_field"])
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(
                    utility, alpha=1 - group["beta_utility"]
                )
                scaled_utility = torch.tanh_(
                    F.normalize((avg_utility / bias_correction), dim=-1) / group["temp"]
                )
                p.data.add_(
                    p.grad.data + noise * (1 - scaled_utility), alpha=-group["lr"]
                )

# UPGD: Utilited-based Perturbed Gradient Descent: variation 1 with Anti-correlated noise
class SecondOrderUPGDv1NormalMax(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, method_field=type(self).method.savefield, noise_damping=noise_damping, names=names)
        super(SecondOrderUPGDv1NormalMax, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["max_utility"] = torch.tensor(-torch.inf)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                if group["noise_damping"]:
                    noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                else:
                    noise = torch.randn_like(p.grad) * group["sigma"]
                avg_utility = state["avg_utility"]
                hess_param = getattr(p, group["method_field"])
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(
                    utility, alpha=1 - group["beta_utility"]
                )
                current_max = avg_utility.max()
                if state["max_utility"] < current_max:
                    state["max_utility"] = current_max
                scaled_utility = torch.tanh_((avg_utility / bias_correction) / group["temp"]) / torch.tanh(state["max_utility"])
                p.data.add_(
                    p.grad.data + noise * (1 - scaled_utility), alpha=-group["lr"]
                )
class SecondOrderUPGDv2AntiCorrNormalized(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, method_field=type(self).method.savefield, noise_damping=noise_damping, names=names)
        super(SecondOrderUPGDv2AntiCorrNormalized, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                if group["noise_damping"]:
                    noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                else:
                    noise = torch.randn_like(p.grad) * group["sigma"]
                avg_utility = state["avg_utility"]
                hess_param = getattr(p, group["method_field"])
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(
                    utility, alpha=1 - group["beta_utility"]
                )
                scaled_utility = torch.tanh_(
                    F.normalize((avg_utility / bias_correction), dim=-1) / group["temp"]
                )
                p.data.add_(
                    (p.grad.data + noise) * (1 - scaled_utility), alpha=-group["lr"]
                )

# UPGD: Utilited-based Perturbed Gradient Descent: variation 2 with Anti-correlated noise
class SecondOrderUPGDv2AntiCorrMax(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, method_field=type(self).method.savefield, noise_damping=noise_damping, names=names)
        super(SecondOrderUPGDv2AntiCorrMax, self).__init__(params, defaults)
    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["max_utility"] = torch.tensor(-torch.inf)
                state["step"] += 1
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                if group["noise_damping"]:
                    noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                else:
                    noise = torch.randn_like(p.grad) * group["sigma"]
                avg_utility = state["avg_utility"]
                hess_param = getattr(p, group["method_field"])
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(
                    utility, alpha=1 - group["beta_utility"]
                )
                current_max = avg_utility.max()
                if state["max_utility"] < current_max:
                    state["max_utility"] = current_max
                scaled_utility = torch.tanh_((avg_utility / bias_correction) / group["temp"]) / torch.tanh(state["max_utility"])
                p.data.add_(
                    (p.grad.data + noise) * (1 - scaled_utility), alpha=-group["lr"]
                )