import torch
from .task import Task


class SummerWithSignChange(Task):
    """
    Iteretable task that sums a number of random signals and changes target sign every certain number of steps.
    """

    def __init__(
        self,
        name="summer_with_sign_change",
        n_inputs=16,
        n_outputs=1,
        n_operands=2,
        batch_size=32,
        change_freq=100,
    ):
        super().__init__(name, batch_size)
        self.change_freq = change_freq
        self.multiplier = 1
        self.step = 0
        self.criterion = "mse"
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.signals = torch.randint(0, n_inputs, (n_operands,))

    def __next__(self):
        self.step += 1
        if self.step % self.change_freq == 0:
            self.change_sign()

        try:
            return next(self.iterator)
        except StopIteration:
            raise StopIteration()

    def generator(self):
        while True:
            inputs = torch.randn((self.batch_size, self.n_inputs))
            yield inputs, self.multiplier * inputs[:, self.signals].sum(
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
