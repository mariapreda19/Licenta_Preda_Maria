import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np
import cv2

IMG_PATH: str = "../datasets-Error_Analysis/Test/golden_retirever/n02099601_1259.jpg"
MODEL_PATH: str = "model_grad_cam_plusPlus.pth"
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_net(path: str) -> nn.Module:
    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, 6)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    return net.to(DEVICE).eval()

def store_activations() -> tuple:
    captured = {"acts": None, "grads": None}

    def fwd_hook(mod, inp, outp):
        captured["acts"] = outp.detach()

    def bwd_hook(mod, grad_inp, grad_outp):
        captured["grads"] = grad_outp[0].detach()

    return fwd_hook, bwd_hook, captured

def prep_input(img_path: str) -> torch.Tensor:
    tr = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return tr(read_image(img_path)).unsqueeze(0).to(DEVICE)

def compute_gradcam_plus_plus(acts: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
    eps = 1e-8
    A = acts[0]
    G = grads[0]
    G2, G3 = G ** 2, G ** 3
    S = A.sum(dim=(1, 2), keepdim=True)
    D = 2 * G2 + S * G3
    D = torch.where(D != 0.0, D, torch.tensor(eps, device=DEVICE))
    alpha = G2 / D
    w = (alpha * F.relu(G)).sum(dim=(1, 2))
    cam = (w[:, None, None] * A).sum(dim=0)
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() + eps)
    return cam.cpu().numpy()

def get_bounding_box(cam: np.ndarray) -> tuple:
    heat = cv2.resize(cam, (224, 224))
    bin_mask = (heat * 255).astype(np.uint8)
    _, th = cv2.threshold(bin_mask, 80, 255, cv2.THRESH_BINARY)
    k = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    c, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if c:
        x, y, w, h = cv2.boundingRect(max(c, key=cv2.contourArea))
        return x, y, w, h
    return 0, 0, 0, 0

def visualize(orig: torch.Tensor, cam: np.ndarray, bbox: tuple) -> None:
    img = cv2.resize(orig.permute(1, 2, 0).numpy(), (224, 224))
    vis_img = np.uint8(255 * img).copy()
    x, y, w, h = bbox
    cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("Heatmap")
    plt.imshow(cv2.resize(cam, (224, 224)), cmap='jet')
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title("BBox")
    plt.imshow(vis_img)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def run_pipeline() -> None:
    model = load_net(MODEL_PATH)
    hook_fwd, hook_bwd, store = store_activations()
    handle_fwd = model.layer4.register_forward_hook(hook_fwd)
    handle_bwd = model.layer4.register_backward_hook(hook_bwd)
    img_tensor = prep_input(IMG_PATH)
    raw_img = read_image(IMG_PATH).float() / 255.0
    out = model(img_tensor)
    cls = out.argmax(dim=1).item()
    model.zero_grad()
    out[0, cls].backward()
    cam = compute_gradcam_plus_plus(store["acts"], store["grads"])
    box = get_bounding_box(cam)
    visualize(raw_img, cam, box)
    handle_fwd.remove()
    handle_bwd.remove()

run_pipeline()
