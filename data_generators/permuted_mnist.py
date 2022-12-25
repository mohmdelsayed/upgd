import torch
import torchvision
import numpy as np 

class PermutedMNIST:
    """
    Iteratable Permuted MNIST dataset

    Each sample is transformed and flattened.
    """

    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.permute()

    def __next__(self):
        try:
            # Samples from dataset
            return next(self.iterator)
        except StopIteration:
            # restart the iterator if the previous iterator is exhausted.
            self.iterator = iter(self.get_dataloader(self.get_dataset(True)))
            return next(self.iterator)

    def __iter__(self):
        return self

    def get_dataset(self, train=True):
        return torchvision.datasets.MNIST(
            "dataset",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    self.permute_transform,
                    torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
                ]
            ),
        )

    def get_dataloader_all(self):
        datasets = []
        for elem in [True, False]:
            dataset = self.get_dataset(elem)
            datasets.append(dataset)
        whole_mnist = torch.utils.data.ConcatDataset(datasets)
        return torch.utils.data.DataLoader(
            whole_mnist,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def get_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def permute(self):
        rng = np.random.default_rng()
        idx = rng.permutation(784)
        self.permute_transform = torchvision.transforms.Lambda(lambda x: x.view(-1)[idx])
        self.dataset = self.get_dataset(True)
        self.iterator = iter(self.get_dataloader(self.dataset))
