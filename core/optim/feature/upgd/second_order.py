import torch, sys, os
from torch.nn import functional as F
sys.path.insert(1, os.getcwd())
from HesScale.hesscale import HesScale


class SecondOrderNonprotectingLocalUPGD(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        self.gate_utility= None
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names, method_field=type(self).method.savefield)
        super(SecondOrderNonprotectingLocalUPGD, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    hess_param = getattr(p, group["method_field"])
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data + 0.5 * hess_param * p.data ** 2, alpha=1 - group["beta_utility"]
                    )
                    self.gate_utility = torch.sigmoid_(F.normalize(avg_utility / bias_correction, dim=-1))
                    continue
                if self.gate_utility is not None:
                    noise = torch.randn_like(p.grad) * group["sigma"]
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(p.grad.data + noise * (1-self.gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(p.grad.data + noise * (1-self.gate_utility.T), alpha=-group["lr"])
                        self.gate_utility = None
                else:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(p.grad.data, alpha=-group["lr"])


class SecondOrderLocalUPGD(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        self.gate_utility= None
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names, method_field=type(self).method.savefield)
        super(SecondOrderLocalUPGD, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    hess_param = getattr(p, group["method_field"])
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data + 0.5 * hess_param * p.data ** 2, alpha=1 - group["beta_utility"]
                    )
                    self.gate_utility = torch.sigmoid_(F.normalize(avg_utility / bias_correction, dim=-1))
                    continue
                if self.gate_utility is not None:
                    noise = torch.randn_like(p.grad) * group["sigma"]
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_((p.grad.data + noise) * (1-self.gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_((p.grad.data + noise) * (1-self.gate_utility.T), alpha=-group["lr"])
                        self.gate_utility = None
                else:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(p.grad.data, alpha=-group["lr"])

class SecondOrderNonprotectingGlobalUPGD(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names, method_field=type(self).method.savefield)
        super(SecondOrderNonprotectingGlobalUPGD, self).__init__(params, defaults)

    def step(self, loss):
        gate_utility = None
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    hess_param = getattr(p, group["method_field"])
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data + 0.5 * hess_param * p.data ** 2, alpha=1 - group["beta_utility"]
                    )
                    current_util_max = avg_utility.max()
                    if current_util_max > global_max_util:
                        global_max_util = current_util_max

        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if 'gate' in name:
                    gate_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)
                    continue
                if gate_utility is not None:
                    noise = torch.randn_like(p.grad) * group["sigma"]
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(p.grad.data + noise * (1-gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(p.grad.data + noise * (1-gate_utility.T), alpha=-group["lr"])
                        gate_utility = None
                else:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(p.grad.data, alpha=-group["lr"])


class SecondOrderGlobalUPGD(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, names=names, method_field=type(self).method.savefield)
        super(SecondOrderGlobalUPGD, self).__init__(params, defaults)

    def step(self, loss):
        gate_utility = None
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    hess_param = getattr(p, group["method_field"])
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data + 0.5 * hess_param * p.data ** 2, alpha=1 - group["beta_utility"]
                    )
                    current_util_max = avg_utility.max()
                    if current_util_max > global_max_util:
                        global_max_util = current_util_max

        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if 'gate' in name:
                    gate_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)
                    continue
                if gate_utility is not None:
                    noise = torch.randn_like(p.grad) * group["sigma"]
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_((p.grad.data + noise) * (1-gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_((p.grad.data + noise) * (1-gate_utility.T), alpha=-group["lr"])
                        gate_utility = None
                else:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(p.grad.data, alpha=-group["lr"])