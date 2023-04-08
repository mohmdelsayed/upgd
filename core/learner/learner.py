from backpack import extend
import torch
class Learner:
    def __init__(self, name, network, optimizer, optim_kwargs, extend=False):
        self.network_cls = network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optim_kwargs = optim_kwargs
        for k, v in optim_kwargs.items():
            if isinstance(v, str):
                optim_kwargs[k] = float(v)
        self.optimizer = optimizer
        self.name = name
        self.extend = extend

    def __str__(self) -> str:
        return self.name

    def predict(self, input):
        output = self.network(input)
        return output

    def set_task(self, task):
        if self.extend:
            self.network = extend(self.network_cls(n_obs=task.n_inputs, n_outputs=task.n_outputs).to(self.device))
        else:
            self.network = self.network_cls(n_obs=task.n_inputs, n_outputs=task.n_outputs).to(self.device)
        self.parameters = self.network.named_parameters()
