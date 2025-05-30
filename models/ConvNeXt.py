import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # self.norm = nn.LayerNorm([dim, 1, 1])
        # self.norm = nn.GroupNorm(1, dim)
        self.norm = nn.LayerNorm(dim)

        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual

class ConvNeXtTiny_NET(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        dims = [96, 192, 384, 768]
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0])
        )

        self.stage1 = self._make_stage(dims[0], 3)
        self.down1 = self._downsample(dims[0], dims[1])
        self.stage2 = self._make_stage(dims[1], 3)
        self.down2 = self._downsample(dims[1], dims[2])
        self.stage3 = self._make_stage(dims[2], 9)
        self.down3 = self._downsample(dims[2], dims[3])
        self.stage4 = self._make_stage(dims[3], 3)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dims[3], num_classes)
        )

    def _make_stage(self, dim, depth):
        return nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(depth)])

    def _downsample(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        return self.head(x)
    
class ConvNeXtProcessor:
    @classmethod
    def train_model(model, train_loader, config):
        model.to(config.device)
        optimizer = config.optimizer_fn(model)
        best_acc = 0.0

        for epoch in range(config.num_epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0
            for x, y in train_loader:
                x, y = x.to(config.device), y.to(config.device)
                optimizer.zero_grad()
                output = model(x)
                loss = config.criterion(output, y)
                loss.backward()

                # add clip-gradient step 
                if config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                optimizer.step()

                total_loss += loss.item()
                correct += (output.argmax(1) == y).sum().item()
                total += y.size(0)

            acc = 100 * correct / total
            print(f"[ConvNeXtTiny] Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.2f}%")

        # save model 
        output_filename = f"{config.out_name}_{config.num_epochs}.pt"
        torch.save(model.state_dict(), output_filename)
        print(f"Model saved to {output_filename}.")
    
            # # Evaluation sau mỗi epoch
            # model.eval()
            # correct_eval, total_eval, loss_eval = 0, 0, 0
            # with torch.no_grad():
            #     for x, y in test_loader:
            #         x, y = x.to(config.device), y.to(config.device)
            #         output = model(x)
            #         loss = config.criterion(output, y)
            #         loss_eval += loss.item()
            #         correct_eval += (output.argmax(1) == y).sum().item()
            #         total_eval += y.size(0)
            # eval_acc = 100 * correct_eval / total_eval
            # eval_loss = loss_eval / len(test_loader)
            # print(f"[ConvNeXtTiny] Evaluation after Epoch {epoch+1}: Loss={eval_loss:.4f}, Accuracy={eval_acc:.2f}%")

            # if eval_acc > best_acc:
            #     best_acc = eval_acc
            #     torch.save(model.state_dict(), config.out_name + ".pt")
            #     print(f"[ConvNeXtTiny] Checkpoint saved at Epoch {epoch+1} with Accuracy={best_acc:.2f}%")
        
    @classmethod
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
        print(f"[ConvNeXtTiny] Test Accuracy: {acc:.2f}% | Avg Loss: {total_loss / len(test_loader):.4f}")
