class Learner:
    def __init__(self, name, network, optimizer, optim_kwargs):
        self.network_cls = network
        self.optim_kwargs = optim_kwargs
        for k, v in optim_kwargs.items():
            if isinstance(v, str):
                optim_kwargs[k] = float(v)
        self.optimizer = optimizer
        self.name = name

    def __str__(self) -> str:
        return self.name

    def predict(self, input):
        output = self.network(input)
        return output

    def set_task(self, task):
        self.network = self.network_cls(n_obs=task.n_inputs, n_outputs=task.n_outputs)
        self.parameters = self.network.parameters()
