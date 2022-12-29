import sys, os
sys.path.insert(1, os.getcwd())
from HesScale.hesscale import HesScale
import torch

class SecondOrderSearchAntiCorrNormalized(torch.optim.Optimizer):
    method = HesScale()
    def __init__(self, params, lr=1e-5, beta_utility=0.0, temp=1.0, sigma=1.0):
        defaults = dict(lr=lr, beta_utility=beta_utility, temp=temp, sigma=sigma, method_field=type(self).method.savefield,
)
        super(SecondOrderSearchAntiCorrNormalized, self).__init__(params, defaults)

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
                scaled_utility = torch.tanh_(
                    (avg_utility / bias_correction)
                    / group["temp"]
                    / state["max_utility"]
                )
                p.data.add_(
                    noise * (1 - scaled_utility), alpha=-group["lr"]
                )

