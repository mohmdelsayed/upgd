import torch
import torchvision
from .task import Task


class TenLabelChangeCIFAR10O(Task):
    """
    Iteratable MNIST task with changing two labels.
    Each sample is a 28x28 image and the label is a number between 0 and 9.
    Two labels are changed every 1000 steps.
    """

    def __init__(self, name="ten_label_change_cifar100", batch_size=1, change_freq=2500):
        self.dataset = self.get_dataset(True)
        self.change_freq = change_freq
        self.step = 0
        self.n_inputs = 3 * 32 * 32
        self.n_outputs = 10
        self.criterion = "cross_entropy"
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

    def change_two_labels(self):
        selected_classes = torch.randperm(100)[:self.n_outputs]
        dataset = self.get_dataset(True)
        dataset.targets = torch.tensor(dataset.targets)
        indicies = torch.isin(dataset.targets, selected_classes)
        map = {x.item(): i for i, x in enumerate(selected_classes)}
        dataset.targets = dataset.targets[indicies]
        dataset.targets = torch.tensor([map[x] for x in dataset.targets.tolist()])
        dataset.data = dataset.data[indicies]
        self.iterator = iter(self.get_dataloader(dataset))

if __name__ == "__main__":
    task = TenLabelChangeCIFAR10O()
    for i, (x, y) in enumerate(task):
        print(x.shape, y.shape)
        if i == 100:
            break
