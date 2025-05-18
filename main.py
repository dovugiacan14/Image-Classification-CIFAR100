import torch
import argparse
from torchsummary import summary
from datasets import get_dataloader
from models import *
from config import CNNConfig

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
