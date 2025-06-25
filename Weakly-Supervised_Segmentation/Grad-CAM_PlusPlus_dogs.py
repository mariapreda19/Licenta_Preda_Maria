import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


def register_gradcam_hooks(model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    activations, gradients = torch.empty(0), torch.empty(0)

    def forward_hook(_: nn.Module, __: torch.Tensor, out: torch.Tensor) -> None:
        nonlocal activations
        activations = out.detach()

    def backward_hook(_: nn.Module, __: torch.Tensor, grad_out: Tuple[torch.Tensor]) -> None:
        nonlocal gradients
        gradients = grad_out[0].detach()

    target_layer = model.layer4
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)
    return activations, gradients


def preprocess_image(img_path: str) -> Tuple[torch.Tensor, np.ndarray]:
    transform = transforms.Compose([
        transforms.Resize((896, 896)),
        transforms.ConvertImageDtype(torch.float32)
    ])
    image = read_image(img_path)
    tensor = transform(image).unsqueeze(0)
    image_np = transform(image).permute(1, 2, 0).cpu().numpy()
    return tensor, image_np


def compute_gradcam_plus_plus(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    activations: torch.Tensor,
    gradients: torch.Tensor
) -> np.ndarray:
    model.zero_grad()
    output = model(input_tensor)
    score = output[0, class_idx]
    score.backward()

    A = activations[0]
    grad = gradients[0]
    grad_sq = grad ** 2
    grad_cu = grad ** 3
    sum_A = A.sum(dim=(1, 2), keepdim=True)
    denom = 2 * grad_sq + sum_A * grad_cu
    denom = torch.where(denom != 0.0, denom, torch.tensor(1e-8, device=grad.device))
    alphas = grad_sq / denom
    weights = (alphas * F.relu(grad)).sum(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * A).sum(dim=0))
    cam_np = cam.detach().cpu().numpy()
    return (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)


def binarize_cam(cam_np: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    return (cam_np > threshold).astype(np.float32)


def refine_with_crf(image: np.ndarray, seed: np.ndarray) -> np.ndarray:
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    h, w = seed.shape
    probs = np.zeros((2, h, w), dtype=np.float32)
    probs[0] = 1.0 - seed
    probs[1] = seed

    crf = DenseCRF2D(w, h, 2)
    crf.setUnaryEnergy(unary_from_softmax(probs))
    feats = create_pairwise_bilateral(sdims=(5, 5), schan=(5, 5, 5), img=image, chdim=2)
    crf.addPairwiseEnergy(feats, compat=10)
    result = crf.inference(5)
    return np.argmax(result, axis=0).reshape((h, w))


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = image.copy()
    masked[mask == 0] = 0
    return masked


def visualize_all(
    original: np.ndarray,
    cam: np.ndarray,
    seed: np.ndarray,
    dilated: np.ndarray,
    refined: np.ndarray,
    final: np.ndarray
) -> None:
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 5, 1)
    plt.title("Original")
    plt.imshow(original)
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.title("Grad-CAM++")
    plt.imshow(cam, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.title("Seed")
    plt.imshow(seed, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.title("CRF")
    plt.imshow(original)
    plt.imshow(refined, cmap="jet", alpha=0.4)
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.title("Masked")
    plt.imshow(final)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    img_path = "../datasets-Error_Analysis/Test/pug/n02110958_12275.jpg"
    model_path = "model_grad_cam_plusPlus.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, num_classes=6, device=device)
    activations, gradients = register_gradcam_hooks(model)
    input_tensor, input_image_np = preprocess_image(img_path)
    input_tensor = input_tensor.to(device)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    cam_np = compute_gradcam_plus_plus(model, input_tensor, pred_class, activations, gradients)

    seed_np = binarize_cam(cam_np)
    seed_resized = F.interpolate(torch.tensor(seed_np).unsqueeze(0).unsqueeze(0), size=(896, 896), mode="bilinear", align_corners=False)
    seed_np = seed_resized.squeeze().cpu().numpy()

    dilated_mask = cv2.dilate((seed_np > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    crf_mask = refine_with_crf(input_image_np, seed_np)
    masked = apply_mask((input_image_np * 255).astype(np.uint8), crf_mask)

    visualize_all(input_image_np, cam_np, seed_np, dilated_mask, crf_mask, masked)


if __name__ == "__main__":
    main()
