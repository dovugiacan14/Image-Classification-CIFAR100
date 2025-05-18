import torch
import torchvision
from torchvision import transforms
from torchvision import datasets as dset


def data_transform_cifar100_train():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4304), (0.2675, 0.2565, 0.2761)),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4304), (0.2675, 0.2565, 0.2761)),
        ]
    )
    return train_transform, valid_transform


def get_dataloader(save_dir):
    train_transform, valid_transform = data_transform_cifar100_train()
    train_data = dset.CIFAR100(
        root=save_dir, train=True, download=True, transform=train_transform
    )
    test_data = dset.CIFAR100(
        root=save_dir, train=False, download=True, transform=valid_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=32, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=32, shuffle=True, num_workers=0
    )
    return train_loader, test_loader
