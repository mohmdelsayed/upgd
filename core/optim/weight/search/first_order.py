import torch
from torch.nn import functional as F


class FirstOrderSearchLocalUncorrelated(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FirstOrderSearchLocalUncorrelated, self).__init__(params, defaults)

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
                noise = torch.randn_like(p.grad) * group["sigma"]
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                p.data.add_(
                    noise
                    * (
                        1
                        - torch.sigmoid_(
                            F.normalize(avg_utility / bias_correction, dim=-1)
                        )
                    ),
                    alpha=-group["lr"],
                )


class FirstOrderSearchLocalAnticorrelated(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FirstOrderSearchLocalAnticorrelated, self).__init__(params, defaults)

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
                        - torch.sigmoid_(
                            F.normalize(avg_utility / bias_correction, dim=-1)
                        )
                    ),
                    alpha=-group["lr"],
                )

class FirstOrderSearchGlobalUncorrelated(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FirstOrderSearchGlobalUncorrelated, self).__init__(params, defaults)

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
                noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)
                p.data.add_(
                    noise
                    * (1 - scaled_utility),
                    alpha=-group["lr"],
                )


class FirstOrderSearchGlobalAnticorrelated(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, beta_utility=0.0, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, beta_utility=beta_utility, sigma=sigma, names=names)
        super(FirstOrderSearchGlobalAnticorrelated, self).__init__(params, defaults)

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
                new_noise = torch.randn_like(p.grad) * group["sigma"]
                noise = new_noise - state["prev_noise"]
                state["prev_noise"] = new_noise
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction) / global_max_util)
                p.data.add_(
                    noise * (1 - scaled_utility), alpha=-group["lr"]
                )
