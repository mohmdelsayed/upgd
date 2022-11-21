import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import SGD
from search_optimizers.first_order import FirstOrderSearchAntiCorr
import torch
import numpy as np
import matplotlib

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 16})
torch.set_default_tensor_type(torch.DoubleTensor)

n_epochs = 100000
n_seeds = 40
step_sizes = [10**-i for i in range(0, 10)]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_units = 5
        n_obs = 2
        n_outputs = 1
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_obs, hidden_units, True),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_units, hidden_units, True),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_units, n_outputs, True),
        )
        nn.init.kaiming_uniform_(self.model[0].weight.data)
        nn.init.zeros_(self.model[0].bias.data)
        nn.init.kaiming_uniform_(self.model[2].weight.data)
        nn.init.zeros_(self.model[2].bias.data)
    def forward(self, x):
        return self.model(x)

optimizers = [SGD, FirstOrderSearchAntiCorr, ]

colors = ['tab:blue', 'tab:orange', "tab:red"]
inputs = torch.Tensor([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]]).unsqueeze(1)

targets = torch.Tensor([[2],
                        [3],
                        [3],
                        [2]]).unsqueeze(1)

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
            optimizer = optim_(net.parameters(), **optim_kwargs)
            for epoch in list(range(n_epochs)):
                epoch_loss = 0.0
                for input, target in zip(inputs, targets):
                    optimizer.zero_grad()
                    output = net(input)
                    loss = criterion(output, target)
                    epoch_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                losses_per_step_size[seed, epoch] = epoch_loss

        all_losses.append(losses_per_step_size)
        final_performs.append(losses_per_step_size.mean(0).mean())


    plt.subplot(2, 1, 1)
    print(final_performs)
    index = np.nanargmin(final_performs)
    means = all_losses[index].mean(0); stds = all_losses[index].std(axis=0) / np.sqrt(n_seeds)
    plt.plot(means, label= str(optim_.__name__) + r" $10^{" + str(int(np.log10(step_sizes[index]))) +  r"}$")
    plt.fill_between(range(n_epochs), means - stds, means + stds, alpha=0.2)
    plt.ylabel('MSE (best stepsize)')
    plt.xlabel('Epoch')
    # plt.ylim([0.0, 10.0]) 
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(step_sizes, final_performs, '-o', label=str(optim_.__name__) )
    plt.ylabel('Average MSE')
    plt.xlabel('Step Size')
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    # plt.ylim([0.0, 1.5])
    plt.legend()

fig.suptitle(f'Exp')
plt.savefig(f'Exp.pdf')  
