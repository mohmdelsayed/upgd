import torch
import torchvision
from .task import Task


class BinarySplitMNIST(Task):
    """
    Iteratable MNIST with binary classification tasks.
    Each sample is a 28x28 image and the label is a number between 0 and 9.
    The binary classifications is changed every 1000 steps.
    """

    def __init__(self, name="binary_split_mnist", batch_size=32, change_freq=1000):
        self.change_freq = change_freq
        self.step = 0
        self.n_inputs = 784
        self.n_outputs = 2
        self.criterion = "cross_entropy"
        self.prev_label1 = None
        self.prev_label2 = None
        super().__init__(name, batch_size)

    def __next__(self):
        if self.step % self.change_freq == 0:
            self.change_classes()
        self.step += 1

        try:
            # Samples from dataset
            return next(self.iterator)
        except StopIteration:
            # restart the iterator if the previous iterator is exhausted.
            self.change_classes()
            return next(self.iterator)

    def generator(self):
        return iter(self.get_dataloader(self.get_dataset(True)))

    def get_dataset(self, train=True):
        return torchvision.datasets.MNIST(
            "dataset",
            train=train,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                    torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
                ]
            ),
        )

    def get_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def change_classes(self):
        while True:
            label_1, label_2 = torch.randint(0, 10, (2,))
            if label_1 != label_2 and label_1 != self.prev_label1 and label_2 != self.prev_label2 and label_1 != self.prev_label2 and label_2 != self.prev_label1:
                break
        selected_classes = torch.tensor([label_1, label_2])
        dataset = self.get_dataset(True)
        indicies = torch.isin(dataset.targets, selected_classes)
        dataset.targets = dataset.targets[indicies]
        dataset.targets[dataset.targets == label_1] = 0
        dataset.targets[dataset.targets == label_2] = 1
        dataset.data = dataset.data[indicies]
        self.iterator = iter(self.get_dataloader(dataset))
        self.prev_label1 = label_1
        self.prev_label2 = label_2

if __name__ == "__main__":
    task = BinarySplitMNIST()
    for i, (x, y) in enumerate(task):
        print(x.shape, y.shape)
        if i == 10:
            break
