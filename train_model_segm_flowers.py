import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from scipy.io import loadmat
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ------------------------------
# Custom Dataset
# ------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label_list, transform=None):
        self.image_dir = image_dir
        self.labels = label_list
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ------------------------------
# Config
# ------------------------------
image_dir = '/kaggle/input/weakly-supervised-segmentation-dataset/dataset_pointing_game/102flowers/jpg'
label_path = '/kaggle/input/weakly-supervised-segmentation-dataset/dataset_pointing_game/imagelabels.mat'
labels = loadmat(label_path)['labels'][0] - 1  # 0-based
num_classes = len(np.unique(labels))

# ------------------------------
# Transforms
# ------------------------------
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# Dataset and Split
# ------------------------------
full_dataset = CustomImageDataset(image_dir, labels, transform=transform_train)
train_len = int(0.75 * len(full_dataset))
val_len = int(0.15 * len(full_dataset))
test_len = len(full_dataset) - train_len - val_len
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])

# Apply test transforms to val/test
val_dataset.dataset.transform = transform_test
test_dataset.dataset.transform = transform_test

# ------------------------------
# Class Weights
# ------------------------------
train_targets = [labels[i] for i in train_dataset.indices]
weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_targets), y=train_targets)
weights_tensor = torch.FloatTensor(weights)

# ------------------------------
# Dataloaders
# ------------------------------
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ------------------------------
# Model: ResNet18
# ------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights='DEFAULT')
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

# ------------------------------
# Training Loop
# ------------------------------
epochs = 30
best_val_acc = 0.0
model_path = 'best_model.pth'

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels_batch in train_loader:
        imgs, labels_batch = imgs.to(device), labels_batch.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        total += labels_batch.size(0)
        correct += predicted.eq(labels_batch).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels_batch in val_loader:
            imgs, labels_batch = imgs.to(device), labels_batch.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            val_total += labels_batch.size(0)
            val_correct += predicted.eq(labels_batch).sum().item()

    val_acc = val_correct / val_total
    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_path)

    print(f"Epoch {epoch+1:02}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# ------------------------------
# Evaluation
# ------------------------------
model.load_state_dict(torch.load(model_path))
model.eval()
test_correct, test_total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels_batch in test_loader:
        imgs, labels_batch = imgs.to(device), labels_batch.to(device)
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        test_total += labels_batch.size(0)
        test_correct += predicted.eq(labels_batch).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())

print(f"\n✅ Final Test Accuracy: {test_correct / test_total:.4f}")

# ------------------------------
# Confusion Matrix
# ------------------------------
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix on Test Set")
plt.tight_layout()
plt.show()
