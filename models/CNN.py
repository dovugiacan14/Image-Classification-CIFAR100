import os
import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from PIL import Image


class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 100)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


class CNNProcessor:
    @staticmethod
    def train_model(model, train_loader, model_config):
        model.to(model_config.device)
        optimizer = model_config.optimizer_fn(model)
        best_model_path = f"{model_config.out_name}_best.pt"
        last_model_path = f"{model_config.out_name}_last.pt"
        # best_model_path = os.path.join(model_config.out_name, "best.pt")
        # last_model_path = os.path.join(model_config.out_name, "last.pt")
        best_accuracy = 0.0

        for epoch in range(model_config.num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(model_config.device), labels.to(
                    model_config.device
                )
                outputs = model(images)
                loss = model_config.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(
                f"Epoch [{epoch+1}/{model_config.num_epochs}], "
                f"Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%"
            )

            # save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}.")

        # save last model
        torch.save(model.state_dict(), last_model_path)
        print(f"Last model saved to {last_model_path}.")

    @staticmethod
    def evaluate(model, test_loader, device="cpu"):
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        loss_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        avg_loss = loss_total / len(test_loader)
        print(f"Test Accuracy: {acc:.2f}% | Avg Loss: {avg_loss:.4f}")

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
        model.to(device)
        model.eval()
        with torch.no_grad():
            output = model(image)
            return output.argmax(dim=1).item()
