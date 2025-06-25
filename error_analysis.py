import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import os


class DogBreedCNN(nn.Module):
    def __init__(self, num_classes):
        super(DogBreedCNN, self).__init__()
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
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)  # <--- dynamic output layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


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

# train_dir = 'datasets-Error_Analysis/Train'
# test_dir = 'datasets-Error_Analysis/Test'
#
# full_train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
# class_names = full_train_dataset.classes
# train_size = int(0.85 * len(full_train_dataset))
# val_size = len(full_train_dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
# val_dataset.dataset_pointing_game.transform = transform_test
#
# test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
#
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = DogBreedCNN(num_classes=len(class_names)).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0003)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)
#
# # Training
# epochs = 10
# best_val_acc = 0.0
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * images.size(0)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
#
#     epoch_loss = running_loss / total
#     epoch_acc = correct / total
#
#     model.eval()
#     val_correct = 0
#     val_total = 0
#     with torch.no_grad():
#         for val_images, val_labels in val_loader:
#             val_images, val_labels = val_images.to(device), val_labels.to(device)
#             val_outputs = model(val_images)
#             _, val_pred = val_outputs.max(1)
#             val_total += val_labels.size(0)
#             val_correct += val_pred.eq(val_labels).sum().item()
#     val_acc = val_correct / val_total
#     scheduler.step(val_acc)
#
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), 'best_model.pt')
#
#     print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
#
# # Load best model and test
# model.load_state_dict(torch.load('best_model.pt'))
# model.eval()
# test_correct = 0
# test_total = 0
# output_txt = "predictions_on_test.txt"
# with open(output_txt, "w") as f:
#     f.write("ImagePath\tTrueClass\tPredictedClass\tConfidence\n")
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(test_loader):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             probs = torch.softmax(outputs, dim=1)
#             confs, preds = probs.max(1)
#             for j in range(images.size(0)):
#                 global_idx = i * batch_size + j
#                 if global_idx >= len(test_dataset.samples):
#                     break
#                 img_path, _ = test_dataset.samples[global_idx]
#                 f.write(f"{img_path}\t{class_names[labels[j]]}\t{class_names[preds[j]]}\t{confs[j].item():.4f}\n")
#             test_total += labels.size(0)
#             test_correct += preds.eq(labels).sum().item()
# print(f"Test Accuracy: {test_correct / test_total:.4f}")
#
# # Confusion Matrix for 4-Class Dataset
# all_preds = []
# all_labels = []
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, preds = torch.max(outputs, 1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#
# cm = confusion_matrix(all_labels, all_preds)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# disp.plot(xticks_rotation=45)
# plt.title("Confusion Matrix - 7 Classes")
# plt.tight_layout()
# plt.show()




# 2
# train_dir = 'datasets-Error_Analysis_2classes/Train'
# test_dir = 'datasets-Error_Analysis_2classes/Test'
#
# # Prepare datasets
# full_train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
# class_names = full_train_dataset.classes
# train_size = int(0.85 * len(full_train_dataset))
# val_size = len(full_train_dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
# val_dataset.dataset_pointing_game.transform = transform_test
# test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
#
# # Data loaders
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)
#
# # Device and model setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = DogBreedCNN(num_classes=len(class_names)).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0003)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
#
# # Training loop
# epochs = 10
# best_val_acc = 0.0
# model_path = 'best_model_2.pt'
#
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * images.size(0)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
#
#     epoch_loss = running_loss / total
#     epoch_acc = correct / total
#
#     # Validation
#     model.eval()
#     val_correct = 0
#     val_total = 0
#     with torch.no_grad():
#         for val_images, val_labels in val_loader:
#             val_images, val_labels = val_images.to(device), val_labels.to(device)
#             val_outputs = model(val_images)
#             _, val_pred = val_outputs.max(1)
#             val_total += val_labels.size(0)
#             val_correct += val_pred.eq(val_labels).sum().item()
#
#     val_acc = val_correct / val_total
#     scheduler.step(val_acc)
#
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), model_path)
#
#     print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
#
# # Load and evaluate best model
# model.load_state_dict(torch.load(model_path))
# model.eval()
# test_correct = 0
# test_total = 0
# output_txt = "predictions_on_test_2_classes.txt"
#
# with open(output_txt, "w") as f:
#     f.write("ImagePath\tTrueClass\tPredictedClass\tConfidence\n")
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(test_loader):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             probs = torch.softmax(outputs, dim=1)
#             confs, preds = probs.max(1)
#             for j in range(images.size(0)):
#                 global_idx = i * batch_size + j
#                 if global_idx >= len(test_dataset.samples):
#                     break
#                 img_path, _ = test_dataset.samples[global_idx]
#                 f.write(f"{img_path}\t{class_names[labels[j]]}\t{class_names[preds[j]]}\t{confs[j].item():.4f}\n")
#             test_total += labels.size(0)
#             test_correct += preds.eq(labels).sum().item()
#
# print(f"Test Accuracy: {test_correct / test_total:.4f}")
#
# # Confusion matrix
# all_preds = []
# all_labels = []
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, preds = torch.max(outputs, 1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#
# cm = confusion_matrix(all_labels, all_preds)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# disp.plot(xticks_rotation=45)
# plt.title("Confusion Matrix - 2 Classes")
# plt.tight_layout()
# plt.show()



# 3

# train_dir = 'datasets-Error_Analysis_2classes/Train'
# test_dir = 'datasets-Error_Analysis_2classes/Test'
# model_path =
#
#
# transform_train = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
#     transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])
#
# transform_test = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5]*3, [0.5]*3)
# ])
#
#
# full_train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
# class_names = full_train_dataset.classes
# train_size = int(0.85 * len(full_train_dataset))
# val_size = len(full_train_dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
# val_dataset.dataset_pointing_game.transform = transform_test
# test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
#
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = models.resnet18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, len(class_names))
# model = model.to(device)
#
#
# targets = [train_dataset.dataset_pointing_game.targets[i] for i in train_dataset.indices]
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets)
# weights_tensor = torch.FloatTensor(class_weights).to(device)
# criterion = nn.CrossEntropyLoss(weight=weights_tensor)
#
#
# optimizer = optim.Adam(model.parameters(), lr=0.0003)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
#
#
# epochs = 25
# best_val_acc = 0.0
#
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0
#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item() * images.size(0)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()
#
#     epoch_loss = running_loss / total
#     epoch_acc = correct / total
#
#     model.eval()
#     val_correct = 0
#     val_total = 0
#     with torch.no_grad():
#         for val_images, val_labels in val_loader:
#             val_images, val_labels = val_images.to(device), val_labels.to(device)
#             val_outputs = model(val_images)
#             _, val_pred = val_outputs.max(1)
#             val_total += val_labels.size(0)
#             val_correct += val_pred.eq(val_labels).sum().item()
#     val_acc = val_correct / val_total
#     scheduler.step(val_acc)
#
#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), model_path)
#
#     print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")
#
#
# model.load_state_dict(torch.load(model_path))
# model.eval()
# test_correct = 0
# test_total = 0
# output_txt = "predictions_on_test_3.txt"
#
# with open(output_txt, "w") as f:
#     f.write("ImagePath\tTrueClass\tPredictedClass\tConfidence\n")
#     with torch.no_grad():
#         for i, (images, labels) in enumerate(test_loader):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             probs = torch.softmax(outputs, dim=1)
#             confs, preds = probs.max(1)
#             for j in range(images.size(0)):
#                 global_idx = i * batch_size + j
#                 if global_idx >= len(test_dataset.samples):
#                     break
#                 img_path, _ = test_dataset.samples[global_idx]
#                 f.write(f"{img_path}\t{class_names[labels[j]]}\t{class_names[preds[j]]}\t{confs[j].item():.4f}\n")
#             test_total += labels.size(0)
#             test_correct += preds.eq(labels).sum().item()
#
# print(f"Test Accuracy: {test_correct / test_total:.4f}")
#
#
# all_preds = []
# all_labels = []
# with torch.no_grad():
#     for images, labels in test_loader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, preds = torch.max(outputs, 1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#
# cm = confusion_matrix(all_labels, all_preds)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# disp.plot(xticks_rotation=45)
# plt.title("Confusion Matrix - Retriever vs Other")
# plt.tight_layout()
# plt.show()



train_dir = 'datasets-Error_Analysis/Train'
test_dir = 'datasets-Error_Analysis/Test'
model_path = 'best_model.pt'


transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

full_train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
class_names = full_train_dataset.classes
print("Class-to-Index Mapping:", full_train_dataset.class_to_idx)

train_size = int(0.85 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
val_dataset.dataset.transform = transform_test
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(targets), y=targets)
weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

epochs = 30
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
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

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            _, val_pred = val_outputs.max(1)
            val_total += val_labels.size(0)
            val_correct += val_pred.eq(val_labels).sum().item()

    val_acc = val_correct / val_total
    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_path)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")

model.load_state_dict(torch.load(model_path))
model.eval()
test_correct = 0
test_total = 0
output_txt = "predictions_on_test.txt"

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

print(f"Test Accuracy: {test_correct / test_total:.4f}")

all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45, cmap='Blues')  # <-- Ensure blue tones
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()