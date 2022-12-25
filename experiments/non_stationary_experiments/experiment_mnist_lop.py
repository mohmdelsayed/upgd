import torch
import matplotlib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from data_generators.infinite_mnist import InfiniteMNIST
from optim.gt.first_order import ExtendedSGD, ExtendedAntiPGD, FirstOrderUPGDv1AntiCorrNormalized, FirstOrderUPGDv2AntiCorrNormalized
from optim.search.first_order import FirstOrderSearchAntiCorr

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 16})
torch.set_default_tensor_type(torch.DoubleTensor)

n_steps = 50000
n_seeds = 1
interval = 100
batch_size = 32
step_sizes = [10**-1] #[10**-i for i in range(0, 6)]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        n_obs = 28*28
        n_outputs = 10
        hidden_units = 300
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_obs, hidden_units, True),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_units, hidden_units // 2, True),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_units // 2, n_outputs, True),
        )
    def forward(self, x):
        return self.model(x)

optimizers = [ExtendedSGD, FirstOrderUPGDv2AntiCorrNormalized]

colors = ['tab:blue', 'tab:orange', "tab:red", "tab:brown", "tab:purple", "tab:cyan", "tab:olive"]

fig, _ = plt.subplots(2, 2, figsize=(16,10), layout="constrained")

for optim_, color in zip(optimizers, colors):
    print(optim_)
    final_performs = []
    all_losses = []
    all_sats = []
    for step_size in step_sizes:
        print(step_size)
        optim_kwargs = {'lr': step_size}
        losses_per_step_size = np.zeros((n_seeds, n_steps))
        for seed in list(range(n_seeds)):
            data_loader = InfiniteMNIST(batch_size=batch_size)
            print("seed", seed)
            step = 0
            torch.manual_seed(seed)
            net = Net()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim_(net.parameters(), **optim_kwargs)
            avg_losses = 0
            for inputs, targets in data_loader:
                optimizer.zero_grad()
                output = net(inputs)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step(loss)
                avg_losses = loss.item()
                losses_per_step_size[seed, step] = avg_losses
                step += 1
                if step == n_steps:
                    break

                if step % interval == 0:
                    data_loader.change_all_lables()

        N = interval
        x = losses_per_step_size.reshape(n_seeds, n_steps // N, N).mean(axis=-1)
        # x = losses_per_step_size
        all_losses.append(x)
        final_performs.append(x.mean(0).mean())


    plt.subplot(2, 1, 1)
    print(final_performs)
    index = np.nanargmin(final_performs)
    means = all_losses[index].mean(0); stds = all_losses[index].std(axis=0) / np.sqrt(n_seeds)
    plt.plot(means, label= str(optim_.__name__) + r" $10^{" + str(int(np.log10(step_sizes[index]))) +  r"}$")
    plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2)
    plt.ylabel('Cross-entropy (best stepsize)')
    plt.xlabel('Bin (100 sample each)')
    plt.ylim(bottom=0.0)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(step_sizes, final_performs, '-o', label=str(optim_.__name__) )
    plt.ylabel('Average MSE')
    plt.xlabel('Step Size')
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    # plt.ylim([0.0, 1.5])
    plt.legend()

plt.savefig(f'lop_mnist.pdf')
