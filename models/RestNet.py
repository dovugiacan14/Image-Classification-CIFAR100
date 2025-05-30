import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out

class ResNet50_NET(nn.Module):
    def __init__(self, block=ResidualBlock, num_classes=100):
        super(ResNet50_NET, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, 3, stride=1)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        self.layer3 = self._make_layer(block, 256, 6, stride=2)
        self.layer4 = self._make_layer(block, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        layers += [block(out_channels, out_channels) for _ in range(1, blocks)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNetProcessor:
    @staticmethod
    def train_model(model, train_loader, model_config):
        model.to(model_config.device)
        optimizer = model_config.optimizer_fn(model)

        for epoch in range(model_config.num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(model_config.device), labels.to(model_config.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = model_config.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{model_config.num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")

            # val_acc = evaluate(model, test_loader, model_config.device, print_log=False)
            # if val_acc > best_acc:
            #     best_acc = val_acc
            #     torch.save(model.state_dict(), f"{model_config.out_name}_best.pt")
            #     print(f"New best model saved with accuracy: {best_acc:.2f}%")
        # save model
        output_filename = f"{model_config.out_name}_{model_config.num_epochs}.pt"
        torch.save(model.state_dict(), output_filename)
        print(f"Model saved to {output_filename}.")
    
    @staticmethod
    def evaluate(model, test_loader, device="cpu", print_log=True):
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        loss_total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss_total += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        avg_loss = loss_total / len(test_loader)
        if print_log:
            print(f"Test Accuracy: {acc:.2f}% | Avg Loss: {avg_loss:.4f}")
        return acc