import torch
from .task import Task


class ChangingAverage(Task):
    """
    Iteretable task that average a number of random inpuys and changes target sign every certain number of steps.
    """

    def __init__(
        self,
        name="changing_average",
        n_inputs=16,
        n_outputs=1,
        n_operands=2,
        batch_size=32,
        change_freq=50,
    ):
        super().__init__(name, batch_size)
        self.change_freq = change_freq
        self.multiplier = 1
        self.step = 0
        self.criterion = "mse"
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_operands = n_operands

    def __next__(self):
        if self.step == 0:
            self.signals = torch.randint(0, self.n_inputs, (self.n_operands,))
        if self.step % self.change_freq == 0:
            self.change_sign()

        self.step += 1
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopIteration()

    def generator(self):
        while True:
            inputs = torch.randn((self.batch_size, self.n_inputs))
            yield inputs, self.multiplier * inputs[:, self.signals].mean(
                dim=-1
            ).unsqueeze(1)

    def change_sign(self):
        self.multiplier *= -1


if __name__ == "__main__":
    task = SummerWithSignChange()
    for i, (x, y) in enumerate(task):
        print(x.shape, y.shape)
        if i == 100:
            break
