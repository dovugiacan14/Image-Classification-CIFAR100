import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image


class BasicCNN:
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # convolution layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.mp = nn.MaxPool2d(2, 2)  # pooling layer
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # fully connected layer
        self.fc2 = nn.Linear(256, 100)

    def forward(self, X):
        x = self.mp(F.relu(self.conv1(x)))
        x = self.mp(F.relu(self.conv2(x)))
        x = self.mp(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def inference(model, image_path, device="cpu"):
        dt_transforms = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4304), (0.2675, 0.2565, 0.2761)
                ),
            ]
        )
        image = Image.open(image_path).convert("RGB")
        image = dt_transforms(image).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output = model(image)
            return output.argmax(dim=1).item()

    @staticmethod
    def evaluate(model, test_dataset, device="cpu"):
        pass
