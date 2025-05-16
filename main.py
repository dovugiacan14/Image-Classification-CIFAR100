import torch
import argparse
from pathlib import Path
from torchsummary import summary
from datasets import download_dataset
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


if __name__ == "__main__":
    data_dir = "dataset/"
    if not Path(data_dir).exists():
        download_dataset(data_dir)
    input_size = (3, 32, 32)
    args = parse_arguments()
    if args.option == 1:
        model = BasicCNN().to(device)
        summary(model, input_size= input_size) # summary model 
        
        

    print(0)
