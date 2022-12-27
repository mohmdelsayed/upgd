import torch
from .task import Task


class UtilityTask(Task):
    """
    Iteretable task that sums a number of random signals and changes target sign every certain number of steps.
    """

    def __init__(
        self,
        name="utility_task",
        n_inputs=16,
        n_outputs=1,
        n_operands=2,
        batch_size=32,
        change_freq=100,
    ):
        super().__init__(name, batch_size)
        self.criterion = "mse_hesscale"
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.signals = torch.randint(0, n_inputs, (n_operands,))

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            raise StopIteration()

    def generator(self):
        while True:
            inputs = torch.rand((self.batch_size, self.n_inputs)) - 0.5
            yield inputs, inputs[:, self.signals].sum(
                dim=-1
            ).unsqueeze(1)


if __name__ == "__main__":
    task = UtilityTask()
    for i, (x, y) in enumerate(task):
        print(x.shape, y.shape)
        if i == 100:
            break
