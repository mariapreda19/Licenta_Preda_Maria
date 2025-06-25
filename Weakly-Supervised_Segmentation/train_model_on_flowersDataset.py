import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from typing import List, Tuple
from scipy.io import loadmat
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class CustomImageDataset(Dataset):
    def __init__(self, image_dir: str, label_list: List[int], transform=None):
        self.image_dir = image_dir
        self.labels = label_list
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def build_datasets(
    image_dir: str, label_mat_path: str, train_ratio: float = 0.75, val_ratio: float = 0.15
) -> Tuple[Dataset, Dataset, Dataset, int, List[int]]:
    labels = loadmat(label_mat_path)['labels'][0] - 1
    dataset = CustomImageDataset(image_dir, labels, transform=transform_train)
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
    val_set.dataset.transform = transform_test
    test_set.dataset.transform = transform_test
    return train_set, val_set, test_set, len(np.unique(labels)), labels


def build_loaders(
    train_set: Dataset, val_set: Dataset, test_set: Dataset, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size)
    )


def compute_class_weights(train_indices: List[int], all_labels: List[int]) -> torch.Tensor:
    train_targets = [all_labels[i] for i in train_indices]
    weights = compute_class_weight('balanced', classes=np.unique(train_targets), y=train_targets)
    return torch.FloatTensor(weights)


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    epochs: int,
    device: torch.device,
    save_path: str
) -> None:
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                val_correct += preds.eq(y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch+1:02}/{epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Val: {val_acc:.4f}")


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, List[int], List[int]]:
    model.eval()
    correct, total = 0, 0
    preds_all, labels_all = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += preds.eq(y).sum().item()
            total += y.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.cpu().numpy())

    return correct / total, preds_all, labels_all


def plot_confusion_matrix(true_labels: List[int], pred_labels: List[int]) -> None:
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main() -> None:
    image_dir = '../dataset_pointing_game/102flowers/jpg'
    label_path = '../dataset_pointing_game/imagelabels.mat'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, val_ds, test_ds, num_classes, all_labels = build_datasets(image_dir, label_path)
    train_loader, val_loader, test_loader = build_loaders(train_ds, val_ds, test_ds)

    class_weights = compute_class_weights(train_ds.indices, all_labels).to(device)
    model = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=30, device=device, save_path="best_model.pth")

    model.load_state_dict(torch.load("../Error-Analysis/best_model_all_classes.pt"))
    test_acc, preds, labels = evaluate_model(model, test_loader, device)
    print(f"\n Final Test Accuracy: {test_acc:.4f}")
    plot_confusion_matrix(labels, preds)


if __name__ == "__main__":
    main()
