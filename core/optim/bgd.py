import torch

class ExtendedBGD(torch.optim.Optimizer):
    def __init__(self, params, mean_eta=1.0, beta=1.0, std_init=0.1):
        '''
        BGD optimizer with single MC sample
        '''
        names, params = zip(*params)
        defaults = dict(mean_eta=mean_eta, std_init=std_init, beta=beta, names=names)
        super(ExtendedBGD, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                state = self.state[p]
                # state initialization
                if len(state) == 0:
                    state["mean_param"] = p.data.clone()
                    state["std_param"] = torch.zeros_like(p.data).add_(group["std_init"])
                    state["eps"] = torch.normal(torch.zeros_like(p.data), 1)
                    state["std_param"] = torch.zeros_like(p.data).add_(group["std_init"])
                    state["grad_avg"] = torch.zeros_like(p.data)
                    state["grad_avg_eps"] = torch.zeros_like(p.data)
                    # p = mean + std * eps
                    p.data.copy_(state["mean_param"].add(state["std_param"].mul(state["eps"])))    
                mean, std = state["mean_param"], state["std_param"]
                state["eps"] = torch.normal(torch.zeros_like(p.data), 1)
                # update mean and std params
                grad_avg, grad_avg_eps = state["grad_avg"], state["grad_avg_eps"]
                grad_avg.mul_(group["beta"]).add_(p.grad.data, alpha=1 - group["beta"])
                grad_avg_eps.mul_(group["beta"]).add_(p.grad.data.mul(state["eps"]), alpha=1 - group["beta"])
                mean.add_(-std.pow(2).mul(grad_avg).mul(group["mean_eta"]))
                sqrt_term = torch.sqrt(grad_avg_eps.mul(std).div(2).pow(2).add(1)).mul(std)
                std.copy_(sqrt_term.add(-grad_avg_eps.mul(std.pow(2)).div(2)))
                # p = mean + std * eps
                p.data.copy_(state["mean_param"].add(state["std_param"].mul(state["eps"])))