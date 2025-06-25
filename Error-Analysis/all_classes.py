import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import utils

data_root = '../datasets-Error_Analysis/Train'
test_root = '../datasets-Error/Test'

model_out = 'best_model_all_classes.pt'
pred_log = 'test_predictions_all_classes.txt'
cm_img = 'confusion_matrix_all_classes.png'

total_epochs = 30
bsz = 32
cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

augment_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

augment_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

base_data = datasets.ImageFolder(root=data_root, transform=augment_train)
train_len = int(0.85 * len(base_data))
part_train, part_val = random_split(base_data, [train_len, len(base_data) - train_len])
part_val.dataset.transform = augment_test

infer_data = datasets.ImageFolder(root=test_root, transform=augment_test)
labels = base_data.classes
n_classes = len(labels)

load_train = DataLoader(part_train, batch_size=bsz, shuffle=True)
load_val = DataLoader(part_val, batch_size=bsz)
load_test = DataLoader(infer_data, batch_size=bsz)

net = utils.init_model(n_classes, cuda_device)
loss_fn = utils.build_loss_fn(part_train, cuda_device)
optim = torch.optim.Adam(net.parameters(), lr=0.0003)
lr_step = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', patience=3, factor=0.5)

top_val = 0
for e in range(total_epochs):
    tr_loss, tr_acc = utils.train_epoch(net, load_train, loss_fn, optim, cuda_device)
    val_acc = utils.evaluate_model(net, load_val, cuda_device)
    lr_step.step(val_acc)

    if val_acc > top_val:
        top_val = val_acc
        torch.save(net.state_dict(), model_out)

    print(f"Epoch {e+1}/{total_epochs} | Loss: {tr_loss:.4f} | Acc: {tr_acc:.4f} | Val: {val_acc:.4f}")

net.load_state_dict(torch.load(model_out))
test_acc, pred_list, true_list = utils.run_inference_and_log(net, load_test, labels, cuda_device, pred_log)
print(f"Test Accuracy: {test_acc:.4f}")
utils.render_confusion_matrix(true_list, pred_list, labels, cm_img)
