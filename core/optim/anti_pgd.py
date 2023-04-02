import torch


class ExtendedAntiPGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, sigma=sigma, names=names)
        super(ExtendedAntiPGD, self).__init__(params, defaults)

    def step(self, loss):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["prev_noise"] = torch.zeros_like(p.data)

                new_noise = torch.randn_like(p.grad) * group["sigma"]
                perturbed_gradient = p.grad + (new_noise - state["prev_noise"])
                state["prev_noise"] = new_noise
                p.data.add_(perturbed_gradient, alpha=-group["lr"])
