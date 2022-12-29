import torch


class ExtendedAntiPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, sigma=1.0):
        defaults = dict(lr=lr, sigma=sigma)
        super(ExtendedAntiPGD, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["prev_noise"] = torch.zeros_like(p.data) * group["sigma"]

                new_noise = torch.randn_like(p.grad)
                perturbed_gradient = p.grad + (new_noise - state["prev_noise"])
                state["prev_noise"] = new_noise
                p.data.add_(perturbed_gradient, alpha=-group["lr"])
