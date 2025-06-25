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
MODEL_PATH: str = "model_grad_cam.pth"
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_model(path: str) -> nn.Module:
    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, 6)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    return net.to(DEVICE).eval()

def grab_hooks() -> tuple:
    data = {"a": None, "g": None}

    def f_hook(m, i, o):
        data["a"] = o.detach()

    def b_hook(m, gi, go):
        data["g"] = go[0].detach()

    return f_hook, b_hook, data

def tensor_input(p: str) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return tf(read_image(p)).unsqueeze(0).to(DEVICE)

def gradcam_core(a: torch.Tensor, g: torch.Tensor) -> np.ndarray:
    w = g.mean(dim=(1, 2))[:, None, None]
    cam = (w * a).sum(dim=0)
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam.cpu().numpy()

def extract_box(c: np.ndarray) -> tuple:
    r = cv2.resize(c, (224, 224))
    m = (r * 255).astype(np.uint8)
    _, b = cv2.threshold(m, 80, 255, cv2.THRESH_BINARY)
    k = np.ones((3, 3), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k, iterations=2)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, k, iterations=1)
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return x, y, w, h
    return 0, 0, 0, 0

def render_view(rgb: torch.Tensor, c: np.ndarray, b: tuple) -> None:
    img = cv2.resize(rgb.permute(1, 2, 0).numpy(), (224, 224))
    disp = np.uint8(255 * img).copy()
    x, y, w, h = b
    cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM")
    plt.imshow(cv2.resize(c, (224, 224)), cmap='jet')
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title("Region")
    plt.imshow(disp)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def execute() -> None:
    net = setup_model(MODEL_PATH)
    h1, h2, cache = grab_hooks()
    reg1 = net.layer4.register_forward_hook(h1)
    reg2 = net.layer4.register_backward_hook(h2)
    inp = tensor_input(IMG_PATH)
    raw = read_image(IMG_PATH).float() / 255.0
    out = net(inp)
    pred = out.argmax(dim=1).item()
    net.zero_grad()
    out[0, pred].backward()
    cam = gradcam_core(cache["a"][0], cache["g"][0])
    box = extract_box(cam)
    render_view(raw, cam, box)
    reg1.remove()
    reg2.remove()

execute()
