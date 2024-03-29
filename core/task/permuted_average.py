import torch
from .task import Task


class PermutedAverage(Task):
    """
    Iteretable task that averages a number of random inputs which change every certain number of steps.
    """

    def __init__(
        self,
        name="permuted_average",
        n_inputs=16,
        n_outputs=1,
        n_subgroups=2,
        batch_size=1,
        change_freq=200,
    ):
        super().__init__(name, batch_size)
        self.change_freq = change_freq
        self.multiplier = -1
        self.step = 0
        self.criterion = "mse"
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.inc = n_inputs // n_subgroups
        self.change_signals()

    def __next__(self):
        if self.step % self.change_freq == 0:
            self.change_signals()
        self.step += 1

        try:
            return next(self.iterator)
        except StopIteration:
            raise StopIteration()

    def generator(self):
        while True:
            inputs = torch.randn((self.batch_size, self.n_inputs))
            yield inputs, inputs[:, self.signals].mean(dim=-1).unsqueeze(1)

    def change_signals(self):
        self.multiplier = (self.multiplier + 1) % self.inc
        self.signals = torch.arange(
            self.multiplier * (self.n_inputs // self.inc),
            (self.multiplier + 1) * (self.n_inputs // self.inc),
        ).type(torch.long)


if __name__ == "__main__":
    task = SummerWithSignalsChange()
    for i, (x, y) in enumerate(task):
        print(x.shape, y.shape)
        if i == 100:
            break
