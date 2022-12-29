import torch

class ExtendedSGD(torch.optim.SGD):
    def __init__(self, params, lr=1e-5):
        _, params = zip(*params)
        super(ExtendedSGD, self).__init__(params, lr=lr)

    def step(self, loss):
        super().step()