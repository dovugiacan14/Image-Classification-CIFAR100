import torch
import argparse
from config import CNNConfig
from torchsummary import summary
from torchvision import transforms
from torchvision import datasets as dset
from models import *


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


def get_dataloader(save_dir):
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
    # Step 1: Load Dataset
    data_dir = "dataset/"
    input_size = (3, 32, 32)
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
