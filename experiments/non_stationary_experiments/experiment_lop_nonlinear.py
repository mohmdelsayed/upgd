import matplotlib.pyplot as plt
import torch.nn as nn
from optim.gt.first_order import ExtendedSGD, FirstOrderUPGDv1AntiCorrNormalized, FirstOrderUPGDv2AntiCorrNormalized
from optim.search.first_order import FirstOrderSearchAntiCorr

import torch
import numpy as np
import matplotlib

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 16})
torch.set_default_tensor_type(torch.DoubleTensor)
n_obs = 16
batch_size = 16
n_epochs = 1000000
n_seeds = 1
freq_change = 100
step_sizes = [10**-2]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_units =  300
        n_outputs = 1
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_obs, hidden_units, True),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_units, hidden_units // 2, True),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_units // 2, n_outputs, True),
        )
        nn.init.kaiming_uniform_(self.model[0].weight)
        nn.init.kaiming_uniform_(self.model[2].weight)
        nn.init.kaiming_uniform_(self.model[4].weight)

    def forward(self, x):
        return self.model(x)

optimizers = [ExtendedSGD]
# optimizers = [ExtendedSGD, ExtendedAntiPGD, FirstOrderUPGDv1AntiCorrNormalized, FirstOrderUPGDv2AntiCorrNormalized, FirstOrderSearchAntiCorr]
colors = ['tab:blue', 'tab:orange', "tab:red", "tab:brown", "tab:purple", "tab:cyan", "tab:olive"]

def func(inputs, mul, inc):
    signals = torch.arange(mul * (n_obs//inc), (mul+1)*(n_obs//inc)).type(torch.long)
    return (inputs[:, signals].sum(dim=-1).unsqueeze(0).T)

inc = n_obs // 2

fig, _ = plt.subplots(2, 2, figsize=(16,10), layout="constrained")

for optim_, color in zip(optimizers, colors):
    print(optim_)
    final_performs = []
    all_losses = []
    all_sats = []
    for step_size in step_sizes:
        print(step_size)
        optim_kwargs = {'lr': step_size}
        losses_per_step_size = np.zeros((n_seeds, n_epochs))
        for seed in list(range(n_seeds)):
            print("seed", seed)
            torch.manual_seed(seed)
            net = Net()
            criterion = nn.MSELoss()
            mul = 0
            optimizer = optim_(net.parameters(), **optim_kwargs)
            for epoch in list(range(n_epochs)):
                inputs = torch.randn((batch_size, n_obs))
                if epoch % freq_change == 0:
                    mul = (mul + 1) % inc

                target = func(inputs, mul, inc)
                optimizer.zero_grad()
                output = net(inputs)
                loss = criterion(output, target)

                epoch_loss = loss.item()
                loss.backward()
                optimizer.step(loss)
                losses_per_step_size[seed, epoch] = epoch_loss

        N = freq_change
        x = losses_per_step_size.reshape(n_seeds, n_epochs // N, N).mean(axis=-1)
        # x = losses_per_step_size
        all_losses.append(x)
        final_performs.append(x.mean(0).mean())

    plt.subplot(2, 1, 1)
    print(final_performs)
    index = np.nanargmin(final_performs)
    means = all_losses[index].mean(0); stds = all_losses[index].std(axis=0) / np.sqrt(n_seeds)
    plt.plot(means, label= str(optim_.__name__) + r" $10^{" + str(int(np.log10(step_sizes[index]))) +  r"}$")
    plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2)
    plt.ylabel('MSE (best stepsize)')
    plt.xlabel(f'Bin ({freq_change} sample each)')
    # plt.ylim([0.0, 1.5]) 
    plt.ylim(bottom=0.0)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(step_sizes, final_performs, '-o', label=str(optim_.__name__) )
    plt.ylabel('Average MSE')
    plt.xlabel('Step Size')
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    plt.legend()

plt.savefig(f'lop_nonlinear.pdf')  
