import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"


class CNNConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    num_epochs = 2
    batch_size = 32
    device = device
    out_name = "cnn_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.Adam(model.parameters(), lr=CNNConfig.learning_rate)


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
            optimizer=ViTConfig.optimizer_fn, 
            T_max=ViTConfig.T_max
        )
