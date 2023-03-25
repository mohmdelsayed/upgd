import torch

class ExtendedShrinkandPerturb(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, decay=0.99, sigma=1.0):
        names, params = zip(*params)
        defaults = dict(lr=lr, sigma=sigma, decay=decay, names=names)
        super(ExtendedShrinkandPerturb, self).__init__(params, defaults)

    def step(self, loss=0.0):
        for group in self.param_groups:
            for name, p in zip(group["names"], group["params"]):
                if 'gate' in name:
                    continue
                p.data.add_(p.grad, alpha=-group["lr"]).mul_(group["decay"]).add_(torch.randn_like(p.grad) * group["sigma"])

if __name__ == '__main__':
    # simple test
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.autograd import Variable

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2, 10)
            self.fc2 = nn.Linear(10, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    net = Net()
    x = Variable(torch.randn(1, 2), requires_grad=True)
    y = Variable(torch.randn(1, 1), requires_grad=False)
    criterion = nn.MSELoss()
    optimizer = ExtendedShrinkandPerturb(net.named_parameters(), lr=0.001, decay=0.1, sigma=0.1)

    for i in range(1000):
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(loss.item())