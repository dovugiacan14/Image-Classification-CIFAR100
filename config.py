import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_optimizer import Lookahead

device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = "checkpoints/"
os.makedirs(save_dir, exist_ok=True)
num_workers = 16


class CNNConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    num_epochs = 2
    batch_size = 32
    device = device
    out_name = "cnn_model"
    os.makedirs(out_name, exist_ok= True)

    @staticmethod
    def optimizer_fn(model):
        return optim.Adam(model.parameters(), lr=CNNConfig.learning_rate)


class ResNetConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    weight_decay = 1e-4
    num_epochs = 100
    batch_size = 32
    device = device
    out_name = save_dir + "ResNet50/resnet50_model"
    num_workers = num_workers

    @staticmethod
    def optimizer_fn(model):
        return optim.AdamW(
            model.parameters(),
            lr=ResNetConfig.learning_rate,
            weight_decay=ResNetConfig.weight_decay,
        )

    @staticmethod
    def scheduler_fn(optimizer):
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=ResNetConfig.num_epochs
        )


class VGGConfig:
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    num_epochs = 2
    learning_rate = 1e-4
    device = device
    out_name = "vgg16_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.Adam(model.parameters(), lr=VGGConfig.learning_rate)


class DenseNetConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    weight_decay = 1e-4
    num_epochs = 100
    batch_size = 32
    device = device
    out_name = save_dir + "DenseNet/densenet121_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.AdamW(
            model.parameters(),
            lr=DenseNetConfig.learning_rate,
            weight_decay=DenseNetConfig.weight_decay,
        )


class EfficientConfig:
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    num_epochs = 2
    learning_rate = 1e-4
    device = device
    out_name = "efficientnetb0_finetune_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.Adam(model.parameters(), lr=EfficientConfig.learning_rate)


class ConvNeXtConfig:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    learning_rate = 3e-4
    num_epochs = 100
    batch_size = 32
    device = device
    out_name = save_dir + "ConvNeXt/convnext_tiny_model"
    grad_clip = 1.0  # gradient clipping value

    @staticmethod
    def optimizer_fn(model):
        base_opt = optim.AdamW(
            model.parameters(), lr=ConvNeXtConfig.learning_rate, weight_decay=1e-4
        )
        return Lookahead(base_opt)


class ViTConfig:
    criterion = nn.CrossEntropyLoss()
    num_epochs = 2
    batch_size = 32
    learning_rate = 2e-5
    weight_decay = 0.01
    T_max = 10
    device = device
    out_name = "ViT_finetune_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.AdamW(model.parameters(), lr=ViTConfig.learning_rate)

    @staticmethod
    def scheduler():
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer=ViTConfig.optimizer_fn, T_max=ViTConfig.T_max
        )


class SwinConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 100
    batch_size = 32
    device = device
    out_name = save_dir + "SwinTransformer/swin_transformer_tiny_model"
    model_name = "microsoft/swin-tiny-patch4-window7-224"

    @staticmethod
    def optimizer_fn(model):
        return optim.AdamW(
            model.parameters(),
            lr=SwinConfig.learning_rate,
            weight_decay=SwinConfig.weight_decay,
        )
