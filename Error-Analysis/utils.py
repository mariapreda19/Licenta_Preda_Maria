import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from torchvision import models
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple


def init_model(n_classes: int, dev: torch.device, freeze_backbone: bool = False) -> nn.Module:
    model = models.resnet18(pretrained=True)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(dev)


def build_loss_fn(split: Subset, dev: torch.device, label_smoothing: float = 0.0) -> nn.Module:
    y = [split.dataset.targets[i] for i in split.indices]
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    return nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(dev), label_smoothing=label_smoothing)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


def run_inference_and_log(model: nn.Module, loader: DataLoader, class_names: List[str],
                          device: torch.device, log_file: str) -> Tuple[float, List[int], List[int]]:
    model.eval()
    predictions, ground_truths = [], []
    correct, total = 0, 0

    with open(log_file, "w") as f:
        f.write("ImagePath\tTrueClass\tPredictedClass\tConfidence\n")
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                out = model(x)
                probs = torch.softmax(out, dim=1)
                conf, pred = probs.max(1)

                for j in range(x.size(0)):
                    index = i * loader.batch_size + j
                    if index >= len(loader.dataset.samples):
                        break
                    path, _ = loader.dataset.samples[index]
                    f.write(f"{path}\t{class_names[y[j]]}\t{class_names[pred[j]]}\t{conf[j]:.4f}\n")

                correct += (pred == y).sum().item()
                total += y.size(0)
                predictions.extend(pred.cpu().numpy())
                ground_truths.extend(y.cpu().numpy())

    return correct / total, predictions, ground_truths


def render_confusion_matrix(y_true: List[int], y_pred: List[int], class_labels: List[str],
                            output_path: str, normalize: bool = False) -> None:
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(xticks_rotation=45, cmap='Blues', values_format=".2f" if normalize else "d")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
