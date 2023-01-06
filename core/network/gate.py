import torch
import torch.nn as nn
from hesscale.core.hesscale_base import BaseModuleHesScale
from hesscale.core.derivatives import LinearDerivativesHesScale

class GateLayer(nn.Module):
    def __init__(self, input_features):
        super(GateLayer, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, input_features), requires_grad=True)
    def forward(self, input):
        return input * self.weight

# HesScale Support:
class GateLayerGrad(BaseModuleHesScale):
    def __init__(self):
        super().__init__(derivatives=LinearDerivativesHesScale(), params=["weight"])
    def weight(self, ext, module, grad_inp, grad_out, backproped):
        equation = "vni,ni->vi"
        return torch.einsum(equation, (backproped[0], module.input0 ** 2))


if __name__ == "__main__":
    from hesscale import HesScale
    from backpack import backpack, extend
    torch.manual_seed(0)
    batch_size = 16
    input_size = 4
    output_size = 2
    n_hidden = 100
    
    extension = HesScale()
    extension.set_module_extension(GateLayer, GateLayerGrad())
    inputs = torch.randn(batch_size, input_size)
    targets = torch.randint(0, output_size, (batch_size,))

    net = nn.Sequential(
        nn.Linear(input_size, n_hidden), 
        nn.ReLU(),
        GateLayer(n_hidden),
        nn.Linear(n_hidden, n_hidden // 2),
        nn.ReLU(),
        GateLayer(n_hidden // 2),
        nn.Linear(n_hidden // 2, output_size)
        )

    lossfunc = torch.nn.CrossEntropyLoss()

    my_module = extend(net)
    lossfunc = extend(lossfunc)
    loss = lossfunc(net(inputs), targets)

    with backpack(extension):
        loss.backward()

    for m in my_module.modules():
        if isinstance(m, GateLayer):
            print(m.weight.shape, m.weight.hesscale.shape)
