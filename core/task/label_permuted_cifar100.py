import torch
import torchvision
from .task import Task
from PIL import Image
import pickle

class CustomCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)
        self.data = None
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target

class LabelPermutedCIFAR100(Task):
    """
    Iteratable CIFAR-100 task with permuted labels.
    Each sample is a 1000-dimensional resnet50-processed image and the label is a number between 0 and 99.
    The labels are permuted every 5000 steps.
    """

    def __init__(self, name="label_permuted_cifar100", batch_size=1, change_freq=5000):
        self.dataset = self.get_dataset(True)
        self.change_freq = change_freq
        self.step = 0
        self.n_inputs = 1000
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
        dataset= CustomCIFAR100(
            "dataset",
            train=train,
            download=True
        )
        # check if the dataset is already processed
        file_name = 'processed_cifar100.pkl'
        try:
            with open(file_name, 'rb') as f:
                dataset.data = pickle.load(f)
            return dataset
        except:
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,), (0.5,)),
                ]
            )
            images = [transform(Image.fromarray(img)) for img in dataset.data]
            # stack all images into a single tensor
            dataset.data = torch.stack(images)
            resnet = torchvision.models.resnet50(pretrained=True)
            for param in resnet.parameters():
                param.requires_grad_(False)
            resnet.eval()
            dataset.data = resnet(dataset.data)
            # save the processed dataset
            with open(file_name, 'wb') as f:
                pickle.dump(dataset.data, f)
            return dataset

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
