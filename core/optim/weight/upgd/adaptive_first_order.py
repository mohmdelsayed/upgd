import torch, math
from torch.nn import functional as F

class AdaptiveUPGDAntiCorrLayerwiseFO(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0, beta1=0.9, beta2=0.999, damping=1e-8):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, beta1=beta1, beta2=beta2, damping=damping, names=names)
        super(AdaptiveUPGDAntiCorrLayerwiseFO, self).__init__(params, defaults)

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
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient^2 values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1


                # perform Adam step
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["beta1"], group["beta2"]
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(
                    p.grad.data ** 2, alpha=1 - beta2
                )
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                denom = ((exp_avg_sq.sqrt()) / math.sqrt(bias_correction2)).add_(
                    group["damping"]
                )

                bias_correction = 1 - group["beta_utility"] ** state["step"]
                new_noise = torch.randn_like(p.grad) * group["sigma"]
                noise = new_noise - state["prev_noise"]
                state["prev_noise"] = new_noise
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                scaled_utility = torch.sigmoid_(
                    F.normalize((avg_utility / bias_correction), dim=-1)
                )
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (exp_avg/denom/bias_correction1 + noise) * (1 - scaled_utility), alpha=-group["lr"]
                )

class AdaptiveUPGDNormalLayerwiseFO(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.0, beta_utility=0.0, sigma=1.0, beta1=0.9, beta2=0.999, damping=1e-8):
        names, params = zip(*params)
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, beta1=beta1, beta2=beta2, damping=damping, names=names)
        super(AdaptiveUPGDNormalLayerwiseFO, self).__init__(params, defaults)
    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient^2 values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["step"] += 1

                # perform Adam step
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["beta1"], group["beta2"]
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(
                    p.grad.data ** 2, alpha=1 - beta2
                )
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                denom = ((exp_avg_sq.sqrt()) / math.sqrt(bias_correction2)).add_(
                    group["damping"]
                )

                bias_correction = 1 - group["beta_utility"] ** state["step"]
                noise = torch.randn_like(p.grad) * group["sigma"]
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                scaled_utility = torch.sigmoid_(
                    F.normalize((avg_utility / bias_correction), dim=-1)
                )
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (exp_avg/denom/bias_correction1 + noise) * (1 - scaled_utility), alpha=-group["lr"]
                )
