import torch
import matplotlib
import numpy as np
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
from data_generators.infinite_mnist import InfiniteMNIST
from torch.nn import functional as F
from backpack import backpack, extend
from HesScale.hesscale import HesScale

matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({'font.size': 16})
torch.set_default_tensor_type(torch.DoubleTensor)

def compute_spearman_rank_coefficient(fo_utility, oracle_utility):
    fo_list = []
    oracle_list = []
    for fo, oracle in zip(fo_utility, oracle_utility):
        oracle_list += list(oracle.ravel().numpy())
        fo_list += list(fo.ravel().numpy())

    overall_count = len(fo_list)
    fo_list = np.argsort(np.asarray(fo_list))
    oracle_list = np.argsort(np.asarray(oracle_list))

    difference = np.sum((fo_list - oracle_list) ** 2)
    coeff = 1 - 6.0 * difference / (overall_count * (overall_count**2-1))
    return coeff

n_steps = 3000
n_seeds = 30
interval = 50
step_sizes = [10**-2]
batch_size = 1
n_classes = 1
n_obs = 5

class NvidiaUtilityFO:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
    def compute_utility(self):
        with torch.no_grad():
            fo_utility_net = []
            for p in  self.network.parameters():
                fo_utility = (p.data * p.grad) ** 2
                # fo_utility = torch.argsort(fo_utility.ravel(), dim=-1).reshape(p.data.shape)
                fo_utility_net.append(fo_utility)
        return fo_utility_net  


class NvidiaUtilitySO:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
    def compute_utility(self):
        with torch.no_grad():
            fo_utility_net = []
            for p in  self.network.parameters():
                fo_utility = (-p.data * p.grad + 0.5 * (p.data ** 2) * p.hesscale) ** 2
                # fo_utility = torch.argsort(fo_utility.ravel(), dim=-1).reshape(p.data.shape)
                fo_utility_net.append(fo_utility)
        return fo_utility_net  

class SecondOrderUtility:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
    def compute_utility(self):
        with torch.no_grad():
            fo_utility_net = []
            for p in  self.network.parameters():
                fo_utility = -p.data * p.grad + 0.5 * (p.data ** 2) * p.hesscale
                # fo_utility = torch.argsort(fo_utility.ravel(), dim=-1).reshape(p.data.shape)
                fo_utility_net.append(fo_utility)
            return fo_utility_net  


class FirstOrderUtility:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
    def compute_utility(self):
        with torch.no_grad():
            fo_utility_net = []
            for p in  self.network.parameters():
                fo_utility = - p.data * p.grad                
                # fo_utility = torch.argsort(fo_utility.ravel(), dim=-1).reshape(p.data.shape)
                fo_utility_net.append(fo_utility)
            return fo_utility_net  


class WeightUtility:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
    def compute_utility(self):
        with torch.no_grad():
            weight_utility_net = []
            for p in  self.network.parameters():
                weight_utility = torch.abs(p.data)
                # weight_utility = torch.argsort(weight_utility.ravel(), dim=-1).reshape(p.data.shape)
                weight_utility_net.append(weight_utility)
            return weight_utility_net  

class OracleUtility:
    def __init__(self, network, criterion):
        self.criterion = criterion
        self.network = network
    def compute_utility(self, original_loss, inputs, targets):
        with torch.no_grad():
            true_utility_net = []
            for p in  self.network.parameters():
                true_utility = torch.zeros_like(p.data)
                for i, value in enumerate(p.ravel()):
                    old_value = value.clone()
                    p.ravel()[i] = 0.0
                    output = self.network(inputs)
                    loss = self.criterion(output, targets)
                    p.ravel()[i] = old_value
                    true_utility.ravel()[i] = loss - original_loss
                # true_utility = torch.argsort(true_utility.ravel(), dim=-1).reshape(p.data.shape)
                true_utility_net.append(true_utility)
            return true_utility_net

fig, _ = plt.subplots(1,1, figsize=(16,10), layout="constrained")
signals = torch.randint(0, n_obs, (1,))

for step_size in step_sizes:
    print(step_size)
    optim_kwargs = {'lr': step_size}
    corr_true_fo = np.zeros((n_seeds, n_steps))
    corr_true_weight = np.zeros((n_seeds, n_steps))
    # corr_true_nvidia_fo = np.zeros((n_seeds, n_steps))
    # corr_true_nvidia_so = np.zeros((n_seeds, n_steps))
    corr_true_so = np.zeros((n_seeds, n_steps))
    losses = np.zeros((n_seeds, n_steps))
    for seed in list(range(n_seeds)):
        print("seed", seed)
        step = 0
        torch.manual_seed(seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(n_obs, 10, True),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 10, True),
            torch.nn.Tanh(),
            torch.nn.Linear(10, n_classes, True),
        )
        criterion = nn.MSELoss()
        optimizer = SGD(net.parameters(), **optim_kwargs)
        extend(net)
        extend(criterion)

        fo_utility_calculator = FirstOrderUtility(net, criterion)
        weight_utility_calculator = WeightUtility(net, criterion)
        # nvidia_calculator_fo = NvidiaUtilityFO(net, criterion)
        # nvidia_calculator_so = NvidiaUtilitySO(net, criterion)
        oracle_calculator = OracleUtility(net, criterion)
        so_calculator = SecondOrderUtility(net, criterion)
        decayed_loss = 0.0
        while True:
            inputs = torch.rand((batch_size, n_obs)) - 0.5
            
            # target_class = torch.tensor(0.5).unsqueeze(0).unsqueeze(0) # (inputs[:, signals].sum(dim=-1).unsqueeze(0).T)
            target_class = torch.tensor(inputs[:, signals].sum(dim=-1)).unsqueeze(0) # (inputs[:, signals].sum(dim=-1).unsqueeze(0).T)
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, target_class)   

            with backpack(HesScale()):
                loss.backward()         
            # loss.backward()         
            optimizer.step()

            fo_utility = fo_utility_calculator.compute_utility()
            weight_utility = weight_utility_calculator.compute_utility()
            # nvidia_utility_fo = nvidia_calculator_fo.compute_utility()
            # nvidia_utility_so = nvidia_calculator_so.compute_utility()
            so_utility = so_calculator.compute_utility()
            oracle_utility = oracle_calculator.compute_utility(loss, inputs, target_class)
        
            coeff_true_fo = compute_spearman_rank_coefficient(fo_utility, oracle_utility)
            coeff_weight = compute_spearman_rank_coefficient(weight_utility, oracle_utility)
            # coeff_nvidia_fo = compute_spearman_rank_coefficient(nvidia_utility_fo, oracle_utility)
            # coeff_nvidia_so = compute_spearman_rank_coefficient(nvidia_utility_so, oracle_utility)
            coef_so = compute_spearman_rank_coefficient(so_utility, oracle_utility)

            corr_true_fo[seed, step] = coeff_true_fo
            corr_true_weight[seed, step] = coeff_weight
            # corr_true_nvidia_fo[seed, step] = coeff_nvidia_fo
            # corr_true_nvidia_so[seed, step] = coeff_nvidia_so
            losses[seed, step] = loss.item()
            corr_true_so[seed, step] = coef_so

            step += 1
            if step == n_steps:
                break
        N = interval
        # corr_true_nvidia_reshaped_fo = corr_true_nvidia_fo.reshape(n_seeds, n_steps // N, N).mean(axis=-1)
        # corr_true_nvidia_reshaped_so = corr_true_nvidia_so.reshape(n_seeds, n_steps // N, N).mean(axis=-1)
        corr_true_fo_reshaped = corr_true_fo.reshape(n_seeds, n_steps // N, N).mean(axis=-1)
        corr_true_weight_reshaped = corr_true_weight.reshape(n_seeds, n_steps // N, N).mean(axis=-1)
        corr_true_so_reshaped = corr_true_so.reshape(n_seeds, n_steps // N, N).mean(axis=-1)
        losses_reshaped = losses.reshape(n_seeds, n_steps // N, N).mean(axis=-1)


means = losses_reshaped.mean(0); stds = losses_reshaped.std(axis=0) / np.sqrt(n_seeds)
plt.plot(means, label= "Loss")
plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2)

means = corr_true_weight_reshaped.mean(0); stds = corr_true_weight_reshaped.std(axis=0) / np.sqrt(n_seeds)
plt.plot(means, label= "Weight Magnitude")
plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2)

# means = corr_true_nvidia_reshaped_fo.mean(0); stds = corr_true_nvidia_reshaped_so.std(axis=0) / np.sqrt(n_seeds)
# plt.plot(means, label= "First-order Nvidia")
# plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2)

# means = corr_true_nvidia_reshaped_so.mean(0); stds = corr_true_nvidia_reshaped_so.std(axis=0) / np.sqrt(n_seeds)
# plt.plot(means, label= "Second-order Nvidia")
# plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2)

means = corr_true_fo_reshaped.mean(0); stds = corr_true_fo_reshaped.std(axis=0) / np.sqrt(n_seeds)
plt.plot(means, label= "First-order Approximaton")
plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2)

means = corr_true_so_reshaped.mean(0); stds = corr_true_so_reshaped.std(axis=0) / np.sqrt(n_seeds)
plt.plot(means, label= "Second-order Approximation")
plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2)

plt.ylabel("Spearman's rank correlation coefficient")
plt.xlabel(f'Bin ({interval} samples each)')
plt.legend()

plt.savefig(f'approx_utility.pdf')

