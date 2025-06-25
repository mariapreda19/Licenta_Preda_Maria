import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms
from torchvision.io import read_image
from typing import Tuple
from pydensecrf.densecrf import DenseCRF2D
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral


def load_model(path: str, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def register_grad_hooks(model: nn.Module) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    activations, gradients = torch.empty(0), torch.empty(0)

    def forward_hook(_: nn.Module, __: torch.Tensor, output: torch.Tensor) -> None:
        nonlocal activations
        activations = output.detach()

    def backward_hook(_: nn.Module, __: torch.Tensor, grad_out: Tuple[torch.Tensor]) -> None:
        nonlocal gradients
        gradients = grad_out[0].detach()

    layer = model.layer4
    layer.register_forward_hook(forward_hook)
    layer.register_full_backward_hook(backward_hook)

    return model, activations, gradients


def preprocess_image(img_path: str) -> Tuple[torch.Tensor, np.ndarray]:
    image = read_image(img_path)
    transform = transforms.Compose([
        transforms.Resize((896, 896)),
        transforms.ConvertImageDtype(torch.float32)
    ])
    image_tensor = transform(image).unsqueeze(0)
    image_np = transform(image).permute(1, 2, 0).numpy()
    return image_tensor, image_np


def generate_gradcam(
    model: nn.Module, input_tensor: torch.Tensor, class_idx: int,
    activations: torch.Tensor, gradients: torch.Tensor
) -> np.ndarray:
    output = model(input_tensor)
    model.zero_grad()
    output[0, class_idx].backward()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
    cam_np = cam.squeeze().cpu().numpy()
    return (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)


def binarize_seed(cam_np: np.ndarray, threshold: float = 0.15) -> torch.Tensor:
    return (torch.from_numpy(cam_np) > threshold).float()


def apply_crf(image: np.ndarray, seed: np.ndarray) -> np.ndarray:
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    h, w = seed.shape
    probs = np.zeros((2, h, w), dtype=np.float32)
    probs[0] = 1.0 - seed
    probs[1] = seed

    crf = DenseCRF2D(w, h, 2)
    unary = unary_from_softmax(probs)
    crf.setUnaryEnergy(unary)
    features = create_pairwise_bilateral(sdims=(5, 5), schan=(5, 5, 5), img=image, chdim=2)
    crf.addPairwiseEnergy(features, compat=10)

    result = crf.inference(5)
    return np.argmax(result, axis=0).reshape((h, w))


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = (image * 255).astype(np.uint8).copy()
    img[mask == 0] = 0
    return img


def visualize_all(
    original: np.ndarray,
    binary_seed: np.ndarray,
    simple_mask: np.ndarray,
    crf_mask: np.ndarray,
    masked_output: np.ndarray
) -> None:
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(binary_seed, cmap="gray")
    plt.title("Seed")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(original)
    plt.imshow(simple_mask, cmap="jet", alpha=0.4)
    plt.title("Dilated Mask")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(original)
    plt.imshow(crf_mask, cmap="jet", alpha=0.4)
    plt.title("CRF Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(masked_output)
    plt.title("Masked Output")
    plt.axis("off")
    plt.show()


def main() -> None:
    img_path = "../datasets-Error_Analysis/Test/pug/n02110958_12275.jpg"
    model_path = "model_grad_cam.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_tensor, image_np = preprocess_image(img_path)
    model = load_model(model_path, num_classes=6, device=device)
    model, activations, gradients = register_grad_hooks(model)

    input_tensor = image_tensor.to(device)
    logits = model(input_tensor)
    predicted_class = logits.argmax(dim=1).item()

    cam_np = generate_gradcam(model, input_tensor, predicted_class, activations, gradients)
    seed_tensor = binarize_seed(cam_np)
    seed_up = F.interpolate(seed_tensor.unsqueeze(0).unsqueeze(0), size=(896, 896), mode="bilinear", align_corners=False)
    seed_np = seed_up.squeeze().cpu().numpy()

    simple_mask = (seed_np > 0.5).astype(np.uint8)
    simple_mask = cv2.dilate(simple_mask, np.ones((5, 5), np.uint8), iterations=1)

    crf_result = apply_crf(image_np, seed_np)
    masked_result = apply_mask(image_np, crf_result)

    visualize_all(image_np, seed_np, simple_mask, crf_result, masked_result)


if __name__ == "__main__":
    main()
