import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(DenseLayer(in_channels, growth_rate))
            in_channels += growth_rate
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(2, 2),
        )

    def forward(self, x):
        return self.layer(x)


class DenseNet121_NET(nn.Module):
    def __init__(self, growth_rate=32, num_classes=100):
        super().__init__()
        num_layers = [6, 12, 24, 16]
        num_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.block1 = DenseBlock(num_channels, num_layers[0], growth_rate)
        num_channels += num_layers[0] * growth_rate
        self.trans1 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        self.block2 = DenseBlock(num_channels, num_layers[1], growth_rate)
        num_channels += num_layers[1] * growth_rate
        self.trans2 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        self.block3 = DenseBlock(num_channels, num_layers[2], growth_rate)
        num_channels += num_layers[2] * growth_rate
        self.trans3 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2

        self.block4 = DenseBlock(num_channels, num_layers[3], growth_rate)
        num_channels += num_layers[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.trans1(self.block1(x))
        x = self.trans2(self.block2(x))
        x = self.trans3(self.block3(x))
        x = self.block4(x)
        x = self.avgpool(F.relu(self.bn(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)
    

class DenseNetProcessor:
    @staticmethod
    def train_model(model, train_loader, config):
        model.to(config.device)
        optimizer = config.optimizer_fn(model)
        # best_acc = 0.0
        for epoch in range(config.num_epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0
            for x, y in train_loader:
                x, y = x.to(config.device), y.to(config.device)
                optimizer.zero_grad()
                output = model(x)
                loss = config.criterion(output, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (output.argmax(1) == y).sum().item()
                total += y.size(0)
            acc = 100 * correct / total
            print(f"[DenseNet121] Epoch {epoch+1}: Train Loss={total_loss:.4f}, Train Acc={acc:.2f}%")

            # Evaluate after each epoch
            # val_acc, val_loss = evaluate(model, test_loader, config)

            # # Save best model
            # if val_acc > best_acc:
            #     best_acc = val_acc
            #     torch.save(model.state_dict(), f"{config.out_name}_best.pt")
            #     print("[DenseNet121] New best model saved.")
        # save model 
        output_filename = f"{config.out_name}_{config.num_epochs}.pt"
        torch.save(model.state_dict(), output_filename)
        print(f"Model saved to {output_filename}.")
    
    @staticmethod
    def evaluate(model, test_loader, config):
        model.to(config.device)
        model.eval()
        correct, total, total_loss = 0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(config.device), y.to(config.device)
                output = model(x)
                loss = config.criterion(output, y)
                total_loss += loss.item()
                correct += (output.argmax(1) == y).sum().item()
                total += y.size(0)
        acc = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        print(f"[DenseNet121] Test Accuracy: {acc:.2f}% | Avg Loss: {avg_loss:.4f}")
        return acc, avg_loss
    

