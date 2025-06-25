import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List


class DogBreedCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def get_dataloaders(train_dir: str, test_dir: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    full_train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_size = int(0.85 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    val_dataset.dataset.transform = transform_test
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader, full_train_dataset.classes


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler._LRScheduler, device: torch.device,
                epochs: int, model_path: str) -> None:
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")


def test_and_log(model: nn.Module, test_loader: DataLoader, class_names: List[str],
                 model_path: str, device: torch.device, output_file: str) -> float:
    model.load_state_dict(torch.load(model_path))
    model.eval()
    correct, total = 0, 0
    with open(output_file, "w") as f:
        f.write("ImagePath\tTrueClass\tPredictedClass\tConfidence\n")
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                confs, preds = probs.max(1)
                for j in range(images.size(0)):
                    idx = i * test_loader.batch_size + j
                    img_path, _ = test_loader.dataset.samples[idx]
                    f.write(f"{img_path}\t{class_names[labels[j]]}\t{class_names[preds[j]]}\t{confs[j].item():.4f}\n")
                correct += (preds == labels).sum().item()
                total += labels.size(0)
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc


def plot_confusion_matrix(model: nn.Module, test_loader: DataLoader,
                          class_names: List[str], device: torch.device, save_path: str) -> None:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title("Confusion Matrix - All Classes")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def main() -> None:
    train_dir = '../datasets-Error_Analysis/Train'
    test_dir = '../datasets-Error_Analysis/Test'
    model_path = '../Error-Analysis/best_model_all_classes.pt'
    log_file = 'predictions_on_test_all_classes.txt'
    conf_path = 'confusion_matrix_segm_flowers.png'
    batch_size = 32
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader, class_names = get_dataloaders(train_dir, test_dir, batch_size)
    model = DogBreedCNN(num_classes=len(class_names)).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    targets = [train_loader.dataset.dataset.targets[i] for i in train_loader.dataset.indices]
    weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, model_path)
    test_and_log(model, test_loader, class_names, model_path, device, log_file)
    plot_confusion_matrix(model, test_loader, class_names, device, conf_path)


if __name__ == "__main__":
    main()
