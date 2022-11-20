import torch
from backpack import backpack, extend
from perturb_optim import PGD, AntiPGD, FirstOrderUGD, SecondOrderUGD, FirstOrderUPGD, SecondOrderUPGD
torch.manual_seed(0)
hidden_units = 10
n_obs = 10
n_outputs = 1
lr = 0.001
T = 100000
batch_size = 1

myOptimizer = FirstOrderUPGD

model = torch.nn.Sequential(
    torch.nn.Linear(n_obs, hidden_units, False),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_units, n_outputs, False),
)

loss_func = torch.nn.MSELoss()
optimizer = myOptimizer(model.parameters(), lr=lr)

extend(model)
extend(loss_func)
avg_loss = 0.0

for i in range(T):
    inputs = torch.randn((batch_size, n_obs))
    signals = [0]
    target = torch.tanh(inputs[:, signals].sum()).unsqueeze(0).unsqueeze(0)
    prediction = model(inputs)

    optimizer.zero_grad()
    loss = loss_func(prediction, target)
    
    try:
        with backpack(optimizer.method):
            loss.backward()
    except:
            loss.backward()

    avg_loss += (loss.item()-avg_loss) * 0.001
    optimizer.step()
    if i % 1000 == 0:
        print(avg_loss)