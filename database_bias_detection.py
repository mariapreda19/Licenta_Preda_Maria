import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2
from typing import List, Tuple

def check_images(directory: str) -> List[str]:
    corrupt_files: List[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.verify()
                except Exception:
                    corrupt_files.append(path)
    return corrupt_files

def convert_images_to_jpg(directory: str) -> None:
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path).convert('RGB')
                    new_path = os.path.splitext(path)[0] + ".jpg"
                    img.save(new_path, 'JPEG')
                    os.remove(path)
                except Exception as e:
                    print(f"Failed to convert {path}: {e}")


class ClasificareBias(nn.Module):
    def __init__(self) -> None:
        super(ClasificareBias, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


train_dir = 'bias/train'
test_dir = 'bias/test'

convert_images_to_jpg('bias')
bad_images = check_images('bias')
for path in bad_images:
    os.remove(path)

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


full_train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
class_names = full_train_dataset.classes
train_size = int(0.85 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
val_dataset.dataset.transform = transform_test
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ClasificareBias().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)


epochs = 15
best_val_acc = 0.0
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
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = val_correct / val_total
    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model_bias_2.pt')

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")


model.load_state_dict(torch.load('best_model_bias_2.pt'))
model.eval()
test_correct, test_total = 0, 0
output_txt = "predictions.txt"
with open(output_txt, "w") as f:
    f.write("ImagePath\tTrueClass\tPredictedClass\tConfidence\n")
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(1)
            for j in range(images.size(0)):
                global_idx = i * batch_size + j
                if global_idx >= len(test_dataset.samples):
                    break
                img_path, _ = test_dataset.samples[global_idx]
                f.write(f"{img_path}\t{class_names[labels[j]]}\t{class_names[preds[j]]}\t{confs[j].item():.4f}\n")
            test_total += labels.size(0)
            test_correct += preds.eq(labels).sum().item()
print(f"\n Test Accuracy: {test_correct / test_total:.4f}")


class GradCam:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def save_gradient(self, module: nn.Module, grad_input: Tuple, grad_output: Tuple[torch.Tensor]) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[:, target_class].sum()
        loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap = cv2.GaussianBlur(heatmap, (7, 7), sigmaX=0)
        heatmap -= heatmap.min()
        heatmap /= (heatmap.max() + 1e-8)
        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
        return heatmap


def apply_heatmap(img_tensor: torch.Tensor, heatmap: np.ndarray) -> np.ndarray:
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5 + 0.5) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlayed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    return overlayed_img


os.makedirs("gradcam_errors", exist_ok=True)
gradcam = GradCam(model, model.conv4[0])

for i, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    preds = torch.argmax(outputs, dim=1)

    for j in range(images.size(0)):
        if preds[j] != labels[j]:
            img_tensor = images[j].unsqueeze(0).detach().clone().to(device)
            img_tensor.requires_grad_()
            heatmap = gradcam.generate(img_tensor, preds[j].item())
            img_path, _ = test_dataset.samples[i * batch_size + j]
            result_img = apply_heatmap(images[j].cpu(), heatmap)
            out_path = os.path.join("gradcam_errors", f"{os.path.basename(img_path)}_pred-{class_names[preds[j]]}_true-{class_names[labels[j]]}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
