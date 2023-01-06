import torch
from torch.nn import functional as F
eps = 1e-4

class FirstOrderSearchAntiCorrNormalized(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        self.gate_utility= None
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderSearchAntiCorrNormalized, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                    state["prev_noise"] = torch.zeros_like(p.data)
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                    )
                    self.gate_utility = torch.tanh(F.normalize(avg_utility / bias_correction, dim=-1) / group["temp"])
                    continue
                if self.gate_utility is not None:
                    if group["noise_damping"]:
                        new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                    else:
                        new_noise = torch.randn_like(p.grad) * group["sigma"]
                    noise = new_noise - state["prev_noise"]
                    state["prev_noise"] = new_noise
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.add_(noise * (1-self.gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.add_(noise * (1-self.gate_utility.T), alpha=-group["lr"])
                        self.gate_utility = None
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])


class FirstOrderSearchAntiCorrMax(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        self.gate_utility= None
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderSearchAntiCorrMax, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                        state["max_utility"] = torch.tensor(-torch.inf)
                    state["prev_noise"] = torch.zeros_like(p.data)
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                    )
                    current_max = avg_utility.max()
                    if state["max_utility"] < current_max:
                        state["max_utility"] = current_max
                    self.gate_utility = torch.tanh((avg_utility / bias_correction) / group["temp"]) / (torch.tanh(state["max_utility"])+eps)
                    continue
                if self.gate_utility is not None:
                    if group["noise_damping"]:
                        new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                    else:
                        new_noise = torch.randn_like(p.grad) * group["sigma"]
                    noise = new_noise - state["prev_noise"]
                    state["prev_noise"] = new_noise
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.add_(noise * (1-self.gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.add_(noise * (1-self.gate_utility.T), alpha=-group["lr"])
                        self.gate_utility = None
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])


class FirstOrderSearchNormalNormalized(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        self.gate_utility= None
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderSearchNormalNormalized, self).__init__(params, defaults)

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
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                    )
                    self.gate_utility = torch.tanh(F.normalize(avg_utility / bias_correction, dim=-1) / group["temp"])
                    continue
                if self.gate_utility is not None:
                    if group["noise_damping"]:
                        noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                    else:
                        noise = torch.randn_like(p.grad) * group["sigma"]
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.add_(noise * (1-self.gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.add_(noise * (1-self.gate_utility.T), alpha=-group["lr"])
                        self.gate_utility = None
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])


class FirstOrderSearchNormalMax(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        self.gate_utility= None
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderSearchNormalMax, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(reversed(group["names"]), reversed(group["params"])):
                state = self.state[p]
                if len(state) == 0:
                    if 'gate' in name:
                        state["avg_utility"] = torch.zeros_like(p.data)
                        state["step"] = 0
                        state["max_utility"] = torch.tensor(-torch.inf)
                if 'gate' in name:
                    state["step"] += 1
                    bias_correction = 1 - group["beta_utility"] ** state["step"]
                    avg_utility = state["avg_utility"]
                    avg_utility.mul_(group["beta_utility"]).add_(
                        -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                    )
                    current_max = avg_utility.max()
                    if state["max_utility"] < current_max:
                        state["max_utility"] = current_max
                    self.gate_utility = torch.tanh((avg_utility / bias_correction) / group["temp"]) / (torch.tanh(state["max_utility"])+eps)
                    continue
                if self.gate_utility is not None:
                    if group["noise_damping"]:
                        noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                    else:
                        noise = torch.randn_like(p.grad) * group["sigma"]
                    if len(p.data.shape) == 1:
                        # handle bias term
                        p.data.add_(noise * (1-self.gate_utility.squeeze(0)), alpha=-group["lr"])
                    else:
                        # handle weight term
                        p.data.add_(noise * (1-self.gate_utility.T), alpha=-group["lr"])
                        self.gate_utility = None
                else:
                    p.data.add_(p.grad.data, alpha=-group["lr"])