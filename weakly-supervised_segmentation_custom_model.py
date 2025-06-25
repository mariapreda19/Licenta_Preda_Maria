# -------------------- [0] IMPORTURI --------------------
import os, cv2, gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
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
            nn.AdaptiveAvgPool2d((14, 14))  # ← was (7, 7); change to (14, 14)
        )

        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



NUM_CLASSES = 102
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomCNN(num_classes=NUM_CLASSES)
# schimba
checkpoint_path = "/kaggle/input/model_flowers/pytorch/default/1/best_model.pth"  # 🔁 Update if needed
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model = model.to(device).eval()


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook()

    def hook(self):
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)

        grads = self.gradients
        activations = self.activations
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        eps = 1e-8
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        alpha_denom = grads_power_2 * 2.0 + sum_activations * grads_power_3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom) * eps)
        alphas = grads_power_2 / alpha_denom
        weights = (alphas * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
        return cam, class_idx

gradcampp = GradCAMPlusPlus(model, model.conv4)

# schimba
IMG_DIR = "/kaggle/input/weakly-supervised-segmentation-dataset/dataset_pointing_game/102flowers/jpg"
MASK_DIR = "/kaggle/input/weakly-supervised-segmentation-dataset/dataset_pointing_game/102segmentations/segmim"
image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
NUM_IMAGES = 100

transform_tensor = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
transform_np = transforms.Resize((224, 224))


def refine_with_crf(img_np, seed_np):
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)
    h, w = seed_np.shape
    probs = np.stack([1.0 - seed_np, seed_np], axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    unary = unary_from_softmax(probs)
    d.setUnaryEnergy(unary)

    img_chw = img_np.transpose(2, 0, 1)
    feats = create_pairwise_bilateral(sdims=(10, 10), schan=(13, 13, 13), img=img_chw, chdim=0)
    d.addPairwiseEnergy(feats, compat=30)
    d.addPairwiseGaussian(sxy=(3, 3), compat=5)

    Q = d.inference(10)
    return np.argmax(Q, axis=0).reshape((h, w))

def compute_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union > 0 else 0.0

# -------------------- [6] LOOP PE IMAGINI --------------------
THRESHOLD = 0.08  # AICI

ious = []
above_75 = 0
failed = []

for i in tqdm(range(1, NUM_IMAGES + 1)):
    try:
        img_path = os.path.join(IMG_DIR, f"image_{i:05d}.jpg")
        mask_path = os.path.join(MASK_DIR, f"segmim_{i:05d}.jpg")
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            failed.append(i)
            continue

        # Imagine și preprocesare
        image_pil = Image.open(img_path).convert("RGB")
        input_tensor = transform_tensor(image_pil).unsqueeze(0).to(device)
        raw_np = np.array(transform_np(image_pil)) / 255.0

        # CAM
        cam, class_idx = gradcampp(input_tensor)
        cam_np = cam.numpy()

        # Prag simplu CAM
        seed = (cam_np > THRESHOLD).astype(np.float32)

        # CRF – revenim la parametrii originali
        def refine_with_crf(img_np, seed_np):
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8)
            h, w = seed_np.shape
            probs = np.stack([1.0 - seed_np, seed_np], axis=0)
            d = dcrf.DenseCRF2D(w, h, 2)
            unary = unary_from_softmax(probs)
            d.setUnaryEnergy(unary)
            img_chw = img_np.transpose(2, 0, 1)
            feats = create_pairwise_bilateral(sdims=(10, 10), schan=(13, 13, 13), img=img_chw, chdim=0)
            d.addPairwiseEnergy(feats, compat=30)
            d.addPairwiseGaussian(sxy=(3, 3), compat=5)
            Q = d.inference(10)
            return np.argmax(Q, axis=0).reshape((h, w))

        seg_crf = refine_with_crf(raw_np, seed)

        # GT
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (224, 224), interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt > 0).astype(np.uint8)

        # Evaluare
        iou = compute_iou(seg_crf, gt_bin)
        ious.append(iou)
        if iou >= 0.75:
            above_75 += 1

        del input_tensor, cam, seg_crf, gt_bin
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"Eroare la imaginea {i}: {e}")
        failed.append(i)


df = pd.DataFrame([{
    "IoU mediu": round(np.mean(ious), 4) if ious else "n/a",
    "Procent IoU ≥ 75%": round(100 * above_75 / len(ious), 2) if ious else "n/a",
    "Imagini eșuate": len(failed)
}])
print(df)
df.to_csv("rezultate_segmentare.csv", index=False)
