import torch
import torchvision
from .task import Task


class LabelPermutedMNISTOffline(Task):
    """
    Iteratable MNIST task with permuted labels.
    Each sample is a 28x28 image and the label is a number between 0 and 9.
    The labels are permuted every 1000 steps.
    """

    def __init__(self, name="label_permuted_mnist_offline", batch_size=1, change_freq=2500):
        self.dataset = self.get_dataset(train=True)
        self.original_targets = self.dataset.targets.clone()
        self.test_targets = self.get_dataset(train=False).targets.clone()
        self.change_freq = change_freq
        self.step = 0
        self.n_inputs = 784
        self.n_outputs = 47
        self.criterion = "cross_entropy"
        super().__init__(name, batch_size)
        self.new_task = False

    def held_out(self, batch_size=1000, shuffle=False):
        test_dataloader = self.get_dataloader(self.get_dataset(train=False), batch_size=batch_size, shuffle=shuffle)
        test_dataloader.dataset.targets = self.old_permutation[self.test_targets]
        return test_dataloader

    def __next__(self):
        if self.step % self.change_freq == 0:
            self.change_all_lables()
            self.new_task = True
        else:
            self.new_task = False
        self.step += 1

        try:
            # Samples from dataset
            return next(self.iterator), self.new_task
        except StopIteration:
            # restart the iterator if the previous iterator is exhausted.
            self.iterator = self.generator()
            return next(self.iterator), self.new_task

    def generator(self):
        return iter(self.get_dataloader(self.dataset, self.batch_size))

    def get_dataset(self, train=True):
        return torchvision.datasets.EMNIST(
            "dataset",
            train=train,
            download=True,
            split="balanced",
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                    torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
                ]
            ),
        )

    def get_dataloader(self, dataset, batch_size=1, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def change_all_lables(self):
        self.old_permutation = None if self.step == 0 else self.permutation
        self.permutation = torch.randperm(self.n_outputs)
        self.dataset.targets = self.permutation[self.original_targets]
        self.iterator = iter(self.get_dataloader(self.dataset, self.batch_size))

if __name__ == "__main__":
    task = LabelPermutedMNISTOffline()
    for i, ((x, y), new_task) in enumerate(task):
        print(x.shape, y.shape)
        if i == 10:
            break
