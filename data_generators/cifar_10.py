import torch
import torchvision


class InfiniteCIFAR10:
    """
    Iteratable Infinite CIFAR-10 dataset for online learning

    Each sample is transformed and flattened.
    """

    def __init__(self, batch_size=64, noise_mean=0.0, noise_std=1.0):
        self.batch_size = batch_size
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.dataset = self.get_dataset(True)
        self.iterator = iter(self.get_dataloader(self.dataset))

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
        return torchvision.datasets.CIFAR10(
            "dataset",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    # torchvision.transforms.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                    # torchvision.transforms.RandomRotation(degrees=45),
                    # torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
                    # self.add_noise,
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

    def change_two_lables(self):
        label_1, label_2 = torch.randint(0, 10, (2,))
        self.dataset.targets[self.dataset.targets == label_1] = -1
        self.dataset.targets[self.dataset.targets == label_2] = label_1
        self.dataset.targets[self.dataset.targets == -1] = label_2
        self.iterator = iter(self.get_dataloader(self.dataset))

    def change_all_lables(self):
        self.dataset.targets = torch.tensor(self.dataset.targets)
        self.dataset.targets = (self.dataset.targets + 1) % 10
        self.iterator = iter(self.get_dataloader(self.dataset))

    def add_noise(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.noise_std + self.noise_mean


class OfflineCIFAR10:
    """
    Iteratable Offline MNIST dataset for online learning

    Each sample is transformed and flattened.
    """

    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.iterator = iter(self.mnist())

    def __next__(self):
        try:
            # Samples from dataset
            return next(self.iterator)
        except StopIteration:
            # restart the iterator if the previous iterator is exhausted.
            self.iterator = iter(self.mnist())
            return next(self.iterator)

    def __iter__(self):
        return self

    def mnist(self):
        dataset = torchvision.datasets.CIFAR10(
            "dataset",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
                ]
            ),
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def add_noise(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.noise_std + self.noise_mean
