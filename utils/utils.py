import torch
import random
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision import datasets as dset

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloader(save_dir):
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
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


def extract_features_from_loader(pretrained_model, dataloader):
    X = []
    y = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            features = pretrained_model(images)
            X.extend(features.cpu().numpy())
            y.extend(labels.numpy())
    return np.array(X), np.array(y)


