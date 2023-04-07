import torch
import torchvision
from .task import Task


class LabelPermutedCIFAR100(Task):
    """
    Iteratable CIFAR-100 task with permuted labels.
    Each sample is a 32x32x3 image and the label is a number between 0 and 100.
    The labels are permuted every 10000 steps.
    """

    def __init__(self, name="label_permuted_cifar100", batch_size=1, change_freq=10000):
        self.dataset = self.get_dataset(True)
        self.change_freq = change_freq
        self.step = 0
        self.n_inputs = 3 * 32 * 32
        self.n_outputs = 100
        self.criterion = "cross_entropy"
        super().__init__(name, batch_size)

    def __next__(self):
        if self.step % self.change_freq == 0:
            self.change_all_lables()
        self.step += 1

        try:
            # Samples from dataset
            return next(self.iterator)
        except StopIteration:
            # restart the iterator if the previous iterator is exhausted.
            self.iterator = self.generator()
            return next(self.iterator)

    def generator(self):
        return iter(self.get_dataloader(self.dataset))

    def get_dataset(self, train=True):
        return torchvision.datasets.CIFAR100(
            "dataset",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                ]
            ),
        )

    def get_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def change_all_lables(self):
        self.dataset.targets = torch.randperm(self.n_outputs)[self.dataset.targets]
        self.iterator = iter(self.get_dataloader(self.dataset))


if __name__ == "__main__":
    task = LabelPermutedCIFAR100()
    for i, (x, y) in enumerate(task):
        print(x.shape, y.shape)
        if i == 10:
            break
