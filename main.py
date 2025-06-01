import time
import torch
import argparse
from torchsummary import summary
from torchvision.models import efficientnet_b0
from transformers import ViTForImageClassification
from utils.utils import set_random_seed, get_dataloader, extract_features_from_loader
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
        start_time = time.time()
        # train model
        CNNProcessor.train_model(
            model=model, train_loader=train_loader, model_config=CNNConfig
        )
        print(f"Time for training: {time.time() - start_time}")
        # evaluate performance
        CNNProcessor.evaluate(model=model, test_loader=test_loader, device=device)

    elif args.option == 2: 
        model = ResNet50_NET().to(device)
        summary(model, input_size== input_size)

        # train model 
        ResNetProcessor.train_model(
            model= model, 
            train_loader= train_loader, 
            model_config= ResNetConfig
        )

        # evluate performance 
        ResNetProcessor.evaluate(model= model, test_loader= test_loader, device= device)

    elif args.option == 3:
        model = VGG16_NET().to(device)
        summary(model, input_size=input_size)

        start_time = time.time()
        # train model
        VGGProcessor.train_model(
            model=model, train_loader=train_loader, model_config=VGGConfig
        )
        print(f"Time for training: {time.time() - start_time}")
        # evaluate performance
        VGGProcessor.evaluate(model=model, test_loader=test_loader, device=device)

    elif args.option == 4:
        model = DenseNet121_NET().to(device)
        summary(model, input_size= input_size)
        
        # train model 
        DenseNetProcessor.train_model(
            model= model, 
            train_loader= train_loader, 
            config= DenseNetConfig
        )

        # evaluate performance 
        DenseNetProcessor.evaluate(model= model, test_loader= test_loader, device= device)

    elif args.option == 5:
        model = efficientnet_b0(pretrained=True)
        summary(model, input_size=input_size)

        # train model
        EFFICIENT_B0.fine_tune(
            model=model, train_loader=train_loader, model_config=EfficientConfig
        )
        # evaluate performance
        EFFICIENT_B0.evaluate(model=model, test_loader=test_loader, device=device)

    elif args.option == 6:
        model = ConvNeXtTiny_NET().to(device)
        summary(model, input_size= input_size)

        # train model 
        ConvNeXtProcessor.train_model(
            model= model, 
            train_loader= train_loader, 
            config= ConvNeXtConfig
        )

        # evaluate performance
        ConvNeXtProcessor.evaluate(
            model= model, 
            test_loader= test_loader, 
            config= ConvNeXtConfig
        )
    
    elif args.option == 7:
        vit_model_name = "google/vit-base-patch16-224"
        model = ViTForImageClassification.from_pretrained(
            vit_model_name, num_labels=100, ignore_mismatched_sizes=True
        )
        summary(model, input_size=input_size)
        # train model
        VisionTransfomers.fine_tune(
            model=model, train_loader=train_loader, model_config=ViTConfig
        )

        # evaluate performance
        VisionTransfomers.evaluate(model=model, test_loader=test_loader, device=device)

    elif args.option == 8: 
        model = SwinTinyWrapper(SwinConfig).to(device)
        print(model)
        
        # train model 
        SwinProcessor.fine_tune(
            model= model, 
            train_loader= train_loader, 
            config= SwinConfig
        )

        # evaluate 
        SwinProcessor.evaluate(
            model= model, 
            test_loader= test_loader, 
            config= SwinConfig
        ) 

    elif args.option == 9:
        svm_model = SVM()
        pretrained_model = svm_model.load_pretrained_model(device= device)
        
        # get dataset 
        X_train, y_train = extract_features_from_loader(pretrained_model, train_loader)
        X_test, y_test = extract_features_from_loader(pretrained_model, test_loader)
        data_train = (X_train, y_train)
        data_test = (X_test, y_test)

        # train model 
        svm_model.train(data_train)

        # evaluate 
        svm_model.evaluate(data_test)
