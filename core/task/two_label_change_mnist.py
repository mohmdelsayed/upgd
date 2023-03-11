import torch
import torchvision
from .task import Task


class TwoLabelChangeMNIST(Task):
    """
    Iteratable MNIST task with changing two labels.
    Each sample is a 28x28 image and the label is a number between 0 and 9.
    Two labels are changed every 1000 steps.
    """

    def __init__(self, name="two_label_change_mnist", batch_size=32, change_freq=500):
        self.dataset = self.get_dataset(True)
        self.change_freq = change_freq
        self.step = 0
        self.n_inputs = 784
        self.n_outputs = 47
        self.criterion = "cross_entropy"
        self.prev_label1 = None
        self.prev_label2 = None
        super().__init__(name, batch_size)

    def __next__(self):
        if self.step % self.change_freq == 0:
            self.change_two_labels()
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

    def get_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def change_two_labels(self):
        while True:
            label_1, label_2 = torch.randint(0, self.n_outputs, (2,))
            if label_1 != label_2 and label_1 != self.prev_label1 and label_2 != self.prev_label2:
                break
        self.dataset.targets[self.dataset.targets == label_1] = -1
        self.dataset.targets[self.dataset.targets == label_2] = label_1
        self.dataset.targets[self.dataset.targets == -1] = label_2
        self.iterator = iter(self.get_dataloader(self.dataset))
        self.prev_label1 = label_1
        self.prev_label2 = label_2

if __name__ == "__main__":
    task = TwoLabelChangeMNIST()
    for i, (x, y) in enumerate(task):
        print(x.shape, y.shape)
        if i == 10:
            break
