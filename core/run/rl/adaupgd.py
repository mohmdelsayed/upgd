import torch

class AdaptiveUPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, weight_decay=0.001, beta_utility=0.999, sigma=0.001, beta1=0.9, beta2=0.999, eps=1e-5):
        defaults = dict(lr=lr, weight_decay=weight_decay, beta_utility=beta_utility, sigma=sigma, beta1=beta1, beta2=beta2, eps=eps)
        super(AdaptiveUPGD, self).__init__(params, defaults)
    def step(self):
        global_max_util = torch.tensor(-torch.inf)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["avg_utility"] = torch.zeros_like(p.data)
                    state["first_moment"] = torch.zeros_like(p.data)
                    state["sec_moment"] = torch.zeros_like(p.data)
                state["step"] += 1
                avg_utility = state["avg_utility"]
                avg_utility.mul_(group["beta_utility"]).add_(
                    -p.grad.data * p.data, alpha=1 - group["beta_utility"]
                )
                first_moment, sec_moment = state["first_moment"], state["sec_moment"]
                first_moment.mul_(group["beta1"]).add_(p.grad.data, alpha=1 - group["beta1"])
                sec_moment.mul_(group["beta2"]).add_(p.grad.data ** 2, alpha=1 - group["beta2"])
                current_util_max = avg_utility.max()
                if current_util_max > global_max_util:
                    global_max_util = current_util_max
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                bias_correction_utility = 1 - group["beta_utility"] ** state["step"]
                bias_correction_beta1 = 1 - group["beta1"] ** state["step"]
                bias_correction_beta2 = 1 - group["beta2"] ** state["step"]
                exp_avg = state["first_moment"] / bias_correction_beta1
                exp_avg_sq = state["sec_moment"] / bias_correction_beta2
                noise = torch.randn_like(p.grad) * group["sigma"]
                scaled_utility = torch.sigmoid_((state["avg_utility"] / bias_correction_utility) / global_max_util)
                p.data.mul_(1 - group["lr"] * group["weight_decay"]).add_(
                    (exp_avg * (1 - scaled_utility)) / (exp_avg_sq.sqrt() + group["eps"]) + noise * (1-scaled_utility),
                    alpha=-2.0*group["lr"],
                )