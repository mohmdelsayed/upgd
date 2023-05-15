import torch

class OnlineEWC(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, lamda=0.01, beta_weight=0.999, beta_fisher=0.999):
        names, params = zip(*params)
        defaults = dict(lr=lr, lamda=lamda, beta_weight=beta_weight, beta_fisher=beta_fisher, names=names)
        super(OnlineEWC, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["weight_trace"] = torch.zeros_like(p.data)
                    state["fisher_trace"] = torch.zeros_like(p.data)
                state["step"] += 1
                weight_trace = state["weight_trace"]
                fisher_trace = state["fisher_trace"]
                weight_trace.mul_(group["beta_weight"]).add_(p.data, alpha=1 - group["beta_weight"])
                fisher_trace.mul_(group["beta_fisher"]).add_(p.grad.data ** 2, alpha=1 - group["beta_fisher"])
                bias_correction_weight = 1 - group["beta_weight"] ** state["step"]
                bias_correction_fisher = 1 - group["beta_fisher"] ** state["step"]
                weight_consolidation = group["lamda"] * fisher_trace * (p.data - weight_trace / bias_correction_weight) / bias_correction_fisher
                p.data.add_(p.grad.data + weight_consolidation, alpha=-group["lr"])