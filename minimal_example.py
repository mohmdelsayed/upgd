import torch
from backpack import backpack, extend
from search_optimizers.first_order import FirstOrderSearchNormal, FirstOrderSearchAntiCorr

torch.manual_seed(0)
hidden_units = 100
n_obs = 2
n_outputs = 1
lr = 0.05
T = 100000
batch_size = 1

myOptimizer = FirstOrderSearchAntiCorr

model = torch.nn.Sequential(
    torch.nn.Linear(n_obs, hidden_units, True),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_units, hidden_units, True),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_units, n_outputs, True),
)

loss_func = torch.nn.MSELoss()
optimizer = myOptimizer(model.parameters(), lr=lr)

inputs = torch.Tensor([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]]).unsqueeze(1)

targets = torch.Tensor([[2],
                        [3],
                        [3],
                        [2]]).unsqueeze(1)

extend(model)
extend(loss_func)
avg_loss = 0.0

for i in range(T):
    # inputs = torch.randn((batch_size, n_obs))
    # signals = [0, 3, 6, 1]
    # target = torch.tanh(inputs[:, signals].sum()).unsqueeze(0).unsqueeze(0)
    # prediction = model(inputs)

    for input, target in zip(inputs, targets):
        optimizer.zero_grad()
        output = model(input)
        loss = loss_func(output, target)
        try:
            with backpack(optimizer.method):
                loss.backward()
        except:
                loss.backward()
        optimizer.step()
    
    avg_loss += (loss.item()-avg_loss) * 0.01
    optimizer.step()
    if i*4 % 100 == 0:
        print("Averaged Loss:", avg_loss)
