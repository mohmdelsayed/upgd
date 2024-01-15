import torch

class OnlineEWCPlus(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, lamda=0.01, beta_weight=0.999, beta_importance=0.999, eps=1e-3):
        names, params = zip(*params)
        defaults = dict(lr=lr, lamda=lamda, beta_weight=beta_weight, beta_importance=beta_importance, eps=eps, names=names)
        super(OnlineEWCPlus, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["weight_trace"] = torch.zeros_like(p.data)
                    state["fisher_trace"] = torch.zeros_like(p.data)
                    state["scores_trace"] = torch.zeros_like(p.data)
                    state["prev_weights"] = torch.zeros_like(p.data)
                state["step"] += 1
                weight_trace = state["weight_trace"]
                scores_trace = state["scores_trace"]
                fisher_trace = state["fisher_trace"]
                weight_trace.mul_(group["beta_weight"]).add_(p.data, alpha=1 - group["beta_weight"])
                score_estimate = (p.grad.data * (state["prev_weights"] - p.data))/(0.5*fisher_trace*(state["prev_weights"]-p.data)**2 + group["eps"])
                # make score zero if negative
                score_estimate = torch.where(score_estimate < 0, torch.zeros_like(score_estimate), score_estimate)
                scores_trace = 0.5 * (score_estimate + scores_trace)
                fisher_trace.mul_(group["beta_importance"]).add_(p.data ** 2, alpha=1 - group["beta_importance"])
                bias_correction_weight = 1 - group["beta_weight"] ** state["step"]
                bias_correction_importance = 1 - group["beta_importance"] ** state["step"]
                weight_consolidation = group["lamda"] * ((fisher_trace/bias_correction_importance) + scores_trace) * (p.data - weight_trace / bias_correction_weight)
                state["prev_weights"] = p.data.clone()
                p.data.add_(p.grad.data + weight_consolidation, alpha=-group["lr"])