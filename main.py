import torch
import random
import argparse
import numpy as np
from torchsummary import summary
from torchvision import transforms
from torchvision import datasets as dset
from models import *
from config import *


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Image-Classification Options.")
    parser.add_argument(
        "--option",
        type=int,
        choices=range(1, 10),
        default=1,
        help="""Type of model: 
            1: Custom Convolutional Neural Network (BasicCNN) 
            2: ResNet34
            3: VGG16
            4: DenseNet121 
            5: EfficientNet
            6: ConvNeXt 
            7: Vision Transformer 
            8: Swin Transformer 
            9: Support Vector Machine with Feature Extraction 
        """,
    )
    return parser.parse_args()


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloader(save_dir):
    # train_transform = transforms.Compose(
    #     [
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5071, 0.4867, 0.4304), (0.2675, 0.2565, 0.2761)),
    #     ]
    # )
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # valid_transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5071, 0.4867, 0.4304), (0.2675, 0.2565, 0.2761)),
    #     ]
    # )

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


if __name__ == "__main__":
    set_random_seed()

    # Step 1: Load Dataset
    data_dir = "dataset/"
    input_size = (3, 224, 224)
    train_loader, test_loader = get_dataloader(data_dir)

    # Step 2: Main Process
    args = parse_arguments()
    if args.option == 1:
        model = BasicCNN().to(device)
        summary(model, input_size=input_size)  # summary model

        # train model
        CNNProcessor.train_model(
            model=model, train_loader=train_loader, model_config=CNNConfig
        )

        # evaluate performance
        CNNProcessor.evaluate(model=model, test_loader=test_loader, device=device)

    elif args.option == 3:
        model = VGG16_NET().to(device)
        summary(model, input_size=input_size)

        # train model
        VGGProcessor.train_model(
            model=model, train_loader=train_loader, model_config=VGGConfig
        )
        # evaluate performance
        VGGProcessor.evaluate(model=model, test_loader=test_loader, device=device)
        pass
    
    elif args.option == 5:
        
        pass 