import torch
from torch.nn import functional as F

# Utility-based Search Optimizers


class FirstOrderSearchNormalNormalized(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderSearchNormalNormalized, self).__init__(params, defaults)

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
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                p.data.add_(
                    noise
                    * (
                        1
                        - torch.tanh_(
                            F.normalize(avg_utility / bias_correction, dim=-1)
                            / group["temp"]
                        )
                    ),
                    alpha=-group["lr"],
                )


class FirstOrderSearchAntiCorrNormalized(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderSearchAntiCorrNormalized, self).__init__(params, defaults)

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
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                p.data.add_(
                    noise
                    * (
                        1
                        - torch.tanh_(
                            F.normalize(avg_utility / bias_correction, dim=-1)
                            / group["temp"]
                        )
                    ),
                    alpha=-group["lr"],
                )

# UPGD: Utilited-based Perturbed Gradient Descent: variation 2 (utility controls gradient)
class FirstOrderSearchNormalMax(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderSearchNormalMax, self).__init__(params, defaults)

    def step(self, loss):
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                state["step"] += 1
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max
                    
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                if group["noise_damping"]:
                    noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                else:
                    noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.tanh_((state["avg_utility"] / bias_correction) / group["temp"] / global_max_util) / torch.tanh_(torch.tensor(1.0))
                p.data.add_(
                    noise
                    * (1 - scaled_utility),
                    alpha=-group["lr"],
                )


# UPGD: Utilited-based Perturbed Gradient Descent: variation 2 with Anti-correlated noise
class FirstOrderSearchAntiCorrMax(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0, noise_damping=True):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, noise_damping=noise_damping, names=names)
        super(FirstOrderSearchAntiCorrMax, self).__init__(params, defaults)

    def step(self, loss):
        global_max_util = torch.tensor(-torch.inf)
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
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max
                    
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                bias_correction = 1 - group["beta_utility"] ** state["step"]
                if group["noise_damping"]:
                    new_noise = torch.randn_like(p.grad) * group["sigma"] * torch.tanh(loss)
                else:
                    new_noise = torch.randn_like(p.grad) * group["sigma"]
                noise = new_noise - state["prev_noise"]
                state["prev_noise"] = new_noise
                scaled_utility = torch.tanh_((state["avg_utility"] / bias_correction) / group["temp"] / global_max_util) / torch.tanh_(torch.tensor(1.0))
                p.data.add_(
                    noise * (1 - scaled_utility), alpha=-group["lr"]
                )
