import torch
from torch.nn import functional as F


class VisionTransfomers:
    @staticmethod
    def fine_tune(model, train_loader, model_config):
        model.to(model_config.device)
        optimizer = model_config.optimizer_fn(model)
        scheduler = model_config.scheduler()
        for epoch in range(model_config.num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(model_config.device)
                labels = labels.to(model_config.device)

                outputs = model(images)
                loss = model_config.criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            avg_loss = running_loss / len(train_loader)
            accuracy = 100 * correct / total
            print(
                f"Epoch [{epoch+1}/{model_config.num_epochs}], Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%"
            )
            scheduler.step()
            
        # save model
        output_filename = f"{model_config.out_name}_{model_config.num_epochs}.pt"
        torch.save(model.state_dict(), output_filename)
        print(f"Model saved to {output_filename}.")

    @staticmethod
    def evaluate(model, test_loader, device= "cpu"):
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