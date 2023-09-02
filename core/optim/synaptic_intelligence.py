import torch

class SynapticIntelligence(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, lamda=0.01, beta_weight=0.999, beta_importance=0.999, eps=1e-3):
        names, params = zip(*params)
        defaults = dict(lr=lr, lamda=lamda, beta_weight=beta_weight, beta_importance=beta_importance, eps=eps, names=names)
        super(SynapticIntelligence, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["weight_trace"] = torch.zeros_like(p.data)
                    state["delta_trace"] = torch.zeros_like(p.data)
                    state["delta_grad_trace"] = torch.zeros_like(p.data)
                    state["init_weights"] = p.data.clone()
                state["step"] += 1
                weight_trace = state["weight_trace"]
                delta_grad_trace = state["delta_grad_trace"]
                delta_trace = state["delta_trace"]
                weight_trace.mul_(group["beta_weight"]).add_(p.data, alpha=1 - group["beta_weight"])
                delta_grad_trace.mul_(group["beta_importance"]).add_(p.grad.data * (state["init_weights"] - p.data), alpha=1 - group["beta_importance"])
                delta_trace.mul_(group["beta_importance"]).add_(state["init_weights"] - p.data, alpha=1 - group["beta_importance"])
                bias_correction_weight = 1 - group["beta_weight"] ** state["step"]
                bias_correction_importance = 1 - group["beta_importance"] ** state["step"]
                fisher_trace = delta_grad_trace.div(bias_correction_importance).div(delta_trace.pow(2).div(bias_correction_importance ** 2).add(group["eps"]))
                weight_consolidation = group["lamda"] * fisher_trace * (p.data - weight_trace / bias_correction_weight)
                p.data.add_(p.grad.data + weight_consolidation, alpha=-group["lr"])