import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.io import read_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# Pentru CRF
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

# ------------------------------
# CONFIG
# ------------------------------
IMG_PATH = "/kaggle/input/cainidataset/datasets-Error_Analysis/Test/golden_retirever/n02099601_1259.jpg"
OUTPUT_DIR = "output_seeds"
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load Model
# ------------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 6)  # adaptează la numărul tău de clase
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device).eval()

# ------------------------------
# Grad-CAM Manual Hooks
# ------------------------------
activations = None
gradients = None

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()

target_layer = model.layer4
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# ------------------------------
# Image Loading and Preprocessing
# ------------------------------
transform = T.Compose([
    T.Resize((896, 896)),
    T.ConvertImageDtype(torch.float32),
])
image = read_image(IMG_PATH)
input_tensor = transform(image).unsqueeze(0).to(device)

# ------------------------------
# Forward + Backward
# ------------------------------
output = model(input_tensor)
pred_class = output.argmax(dim=1).item()

model.zero_grad()
score = output[0, pred_class]
score.backward()

# ------------------------------
# Compute Grad-CAM
# ------------------------------
weights = gradients.mean(dim=(2, 3), keepdim=True)
cam = (weights * activations).sum(dim=1, keepdim=True)
cam = torch.relu(cam).squeeze().cpu()
cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# ------------------------------
# Binarize seed
# ------------------------------
threshold = 0.3
seed_mask = (cam > threshold).float()

# ------------------------------
# Prepare for Display
# ------------------------------
input_img_np = transform(image).permute(1, 2, 0).numpy()
seed_mask_np = seed_mask.numpy()

# ------------------------------
# VARIANTA 1 – Interpolare + dilatare
# ------------------------------
seed_resized = F.interpolate(seed_mask.unsqueeze(0).unsqueeze(0), size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
seed_resized = seed_resized.squeeze().cpu().numpy()

segmented_interp = (seed_resized > 0.5).astype(np.uint8)
segmented_interp = cv2.dilate(segmented_interp, np.ones((5,5), np.uint8), iterations=1)

# ------------------------------
# VARIANTA 2 – CRF refinement
# ------------------------------
def refine_with_crf(img_np, seed_np):
    h, w = seed_np.shape
    img_np = (img_np * 255).astype(np.uint8)

    probs = np.zeros((2, h, w), dtype=np.float32)
    probs[0] = 1.0 - seed_np
    probs[1] = seed_np

    d = dcrf.DenseCRF2D(w, h, 2)
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img_np, chdim=2)
    d.addPairwiseEnergy(feats, compat=10)

    Q = d.inference(5)
    result = np.argmax(Q, axis=0).reshape((h, w))
    return result

input_img_np_uint8 = (input_img_np * 255).astype(np.uint8)
segmented_crf = refine_with_crf(input_img_np_uint8, seed_mask_np)

# ------------------------------
# Vizualizare toate
# ------------------------------
plt.figure(figsize=(15, 6))

plt.subplot(1, 4, 1)
plt.title("Imagine originală")
plt.imshow(input_img_np)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Seed binarizat")
plt.imshow(seed_mask_np, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Segmentare simplă")
plt.imshow(input_img_np)
plt.imshow(segmented_interp, cmap="jet", alpha=0.4)
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Segmentare CRF")
plt.imshow(input_img_np)
plt.imshow(segmented_crf, cmap="jet", alpha=0.4)
plt.axis("off")

plt.tight_layout()
plt.show()

