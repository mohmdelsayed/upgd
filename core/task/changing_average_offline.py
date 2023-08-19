import torch
from .task import Task


class ChangingAverageOffline(Task):
    """
    Iteretable task that average a number of random inpuys and changes target sign every certain number of steps.
    """

    def __init__(
        self,
        name="changing_average_offline",
        n_inputs=16,
        n_outputs=1,
        n_operands=8,
        batch_size=1,
        change_freq=200,
    ):
        super().__init__(name, batch_size)
        self.change_freq = change_freq
        self.multiplier = 1
        self.step = 0
        self.criterion = "mse"
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_operands = n_operands
        self.signals = torch.randint(0, self.n_inputs, (self.n_operands,))
        self.held_out = self.held_out_func()

    # @property
    # def held_out(self):
    #     for _ in range(5):
    #         multiplier = 1 if torch.rand(1) > 0.5 else -1
    #         inputs = torch.randn((self.batch_size, self.n_inputs))
    #         yield (inputs, torch.sin(multiplier * inputs[:, self.signals].sum(dim=-1).unsqueeze(1))), 1 if multiplier == 1 else 0

    def held_out_func(self):
        held_outs = []
        for _ in range(200):
            multiplier = 1 if torch.rand(1) > 0.5 else -1
            inputs =  torch.randn((self.batch_size, self.n_inputs))
            sample = (inputs, multiplier * inputs[:, self.signals].mean(dim=-1).unsqueeze(1)), 1 if multiplier == 1 else 0
            held_outs.append(sample)
        return held_outs

    def __next__(self):
        if self.step % self.change_freq == 0:
            self.change_sign()

        self.step += 1
        try:
            context = 1 if self.multiplier == 1 else 0
            return next(self.iterator), context
        except StopIteration:
            raise StopIteration()

    def generator(self):
        while True:
            inputs = torch.randn((self.batch_size, self.n_inputs))
            yield inputs, self.multiplier * inputs[:, self.signals].mean(dim=-1).unsqueeze(1)

    def change_sign(self):
        self.multiplier *= -1

if __name__ == "__main__":
    task = ChangingAverageOffline()
    
    # Train task
    for i, sample in enumerate(task):
        (x, y), context = sample
        print("Train:", x.shape, y.shape, context)
        if i == 400: break
    # Test held-out
    for (x, y), context in task.held_out:
        print("Held-out:", x.shape, y.shape, context)
