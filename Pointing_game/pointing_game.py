import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm
import glob
from typing import Optional, Tuple


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam


def generate_gradcam_map(input_tensor: torch.Tensor, model: nn.Module, gradcam: GradCAM, class_idx: int, device: torch.device) -> np.ndarray:
    return gradcam(input_tensor, class_idx=class_idx)


def is_hit(gt_mask: np.ndarray, point: Tuple[int, int]) -> bool:
    return gt_mask[point] > 0


def train_model(model: nn.Module, train_loader: torch.utils.data.DataLoader, device: torch.device, num_classes: int, num_epochs: int = 5) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

    return model.eval()


def main() -> None:
    root = 'dataset_pointing_game'
    image_dir = os.path.join(root, '102flowers', 'jpg')
    segmentation_dir = os.path.join(root, '102segmentations', 'segmim')
    seed_dir = os.path.join(root, 'seeds')
    os.makedirs(seed_dir, exist_ok=True)

    labels = loadmat(os.path.join(root, 'imagelabels.mat'))['labels'][0] - 1
    labels = labels[:len(labels) // 3]

    sample_masks = glob.glob(os.path.join(segmentation_dir, '*'))
    print("Sample masks found:", sample_masks[:5])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 102)
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    class FlowerDataset(torch.utils.data.Dataset):
        def __init__(self, image_dir: str, labels: np.ndarray, transform):
            self.image_dir = image_dir
            self.labels = labels
            self.transform = transform

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
            img_path = os.path.join(self.image_dir, f"image_{idx+1:05d}.jpg")
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            return img, self.labels[idx]

    dataset = FlowerDataset(image_dir, labels, transform)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = train_model(model, train_loader, device, num_classes=102)

    gradcam = GradCAM(model, target_layer='features.29')

    hits, misses = 0, 0
    for i, label in tqdm(enumerate(labels), total=len(labels)):
        img_name = f'image_{i + 1:05d}.jpg'
        img_path = os.path.join(image_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        output = model(input_tensor)
        top5_indices = torch.topk(output, 5).indices[0].tolist()

        gt_path = os.path.join(segmentation_dir, f'segmim_{i + 1:05d}.jpg')
        if not os.path.exists(gt_path):
            print(f"[ERROR] Mask file not found: {gt_path}")
            continue

        gt_mask = cv2.imread(gt_path, 0)
        if gt_mask is None:
            print(f"[ERROR] Cannot read segmentation mask (exists but unreadable): {gt_path}")
            continue

        gt_mask = cv2.resize(gt_mask, (224, 224), interpolation=cv2.INTER_NEAREST)

        found_hit = False
        for class_idx in top5_indices:
            cam = generate_gradcam_map(input_tensor, model, gradcam, class_idx, device)
            threshold = 0.3 * cam.max()
            seed = (cam > threshold).astype(np.uint8)
            np.save(os.path.join(seed_dir, f'{img_name.replace(".jpg", f"_class{class_idx}.npy")}'), seed)

            point = np.unravel_index(np.argmax(cam), cam.shape)
            if is_hit(gt_mask, point):
                hits += 1
                found_hit = True
                break
        if not found_hit:
            misses += 1

    accuracy = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    print(f"\nPointing Game Accuracy: {accuracy:.2%}")


if __name__ == '__main__':
    main()
