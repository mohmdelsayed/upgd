from hesscale import HesScale
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

class AntiPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5):
        defaults = dict(lr=lr)
        super(AntiPGD, self).__init__(params, defaults)

    def step(self):
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


# UGD: Utility-regularized Gradient Descent
class FirstOrderUGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, lamda=1.0, beta_utility=0.999, beta_weight=0.999):
        defaults = dict(lr=lr, lamda=lamda, beta_utility=beta_utility, beta_weight=beta_weight)
        super(FirstOrderUGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["avg_weight"] = torch.zeros_like(p.data)

                avg_utility = state["avg_utility"]
                avg_weight = state["avg_weight"]

                utility = - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(utility, alpha=1 - group["beta_utility"])
                avg_weight.mul_(group["beta_weight"]).add_(p.data, alpha=1 - group["beta_weight"])
                p.data.add_(p.grad.data + group["lamda"] * avg_utility * (p.data - avg_weight), alpha=-group["lr"])


# UPGD: Utilited-based Perturbed Gradient Descent
class FirstOrderUPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.99):
        defaults = dict(lr=lr, beta_utility=beta_utility)
        super(FirstOrderUPGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)

                avg_utility = state["avg_utility"]
                utility = - p.grad.data * p.data
                perturbed_gradient = p.grad.data + torch.randn_like(p.grad)
                avg_utility.mul_(group["beta_utility"]).add_(utility, alpha=1 - group["beta_utility"])
                p.data.add_(perturbed_gradient * (1-torch.sigmoid_(utility)), alpha=-group["lr"])


# UGD: Utility-regularized Gradient Descent
class SecondOrderUGD(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, lamda=1.0, beta_utility=0.999, beta_weight=0.999):
        defaults = dict(lr=lr, method_field=type(self).method.savefield, lamda=lamda, beta_utility=beta_utility, beta_weight=beta_weight)
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

                hess_param = getattr(p, group["method_field"]).detach()
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                avg_utility.mul_(group["beta_utility"]).add_(utility, alpha=1 - group["beta_utility"])
                avg_weight.mul_(group["beta_weight"]).add_(p.data, alpha=1 - group["beta_weight"])
                p.data.add_(p.grad.data + group["lamda"] * avg_utility * (p.data - avg_weight), alpha=-group["lr"])

# UPGD: Utilited-based Perturbed Gradient Descent
class SecondOrderUPGD(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.99):
        defaults = dict(lr=lr, beta_utility=beta_utility, method_field=type(self).method.savefield)
        super(SecondOrderUPGD, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["avg_utility"] = torch.zeros_like(p.data)

                avg_utility = state["avg_utility"]
                hess_param = getattr(p, group["method_field"]).detach()
                utility = 0.5 * hess_param * p.data ** 2 - p.grad.data * p.data
                perturbed_gradient = p.grad.data + torch.randn_like(p.grad)
                avg_utility.mul_(group["beta_utility"]).add_(utility, alpha=1 - group["beta_utility"])
                p.data.add_(perturbed_gradient * (1-torch.sigmoid_(utility)), alpha=-group["lr"])