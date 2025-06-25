import os, gc, cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List, Optional
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 102
IMG_DIR = "../dataset_pointing_game/102flowers/jpg"
MASK_DIR = "../dataset_pointing_game/102segmentations/segmim"
NUM_IMAGES = 100
THRESHOLD = 0.2

transform_tensor = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
transform_np = transforms.Resize((224, 224))


def load_model(model_path: str) -> nn.Module:
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, NUM_CLASSES)
    net.load_state_dict(torch.load(model_path, map_location=device))
    return net.to(device).eval()


class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        output = self.model(input_tensor)
        class_idx = class_idx or output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)

        A = self.activations
        G = self.gradients
        eps = 1e-8
        g2, g3 = G ** 2, G ** 3
        sum_activ = A.sum(dim=(2, 3), keepdim=True)
        denom = 2 * g2 + sum_activ * g3
        denom = torch.where(denom != 0, denom, torch.tensor(eps, device=device))
        alpha = g2 / denom
        weights = (alpha * F.relu(G)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
        return cam, class_idx


def refine_with_crf(img_np: np.ndarray, seed_np: np.ndarray) -> np.ndarray:
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


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return intersection / union if union > 0 else 0.0


def process_image(index: int, gradcam: GradCAMPlusPlus) -> Optional[float]:
    img_path = os.path.join(IMG_DIR, f"image_{index:05d}.jpg")
    mask_path = os.path.join(MASK_DIR, f"segmim_{index:05d}.jpg")
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return None

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform_tensor(image).unsqueeze(0).to(device)
    image_np = np.array(transform_np(image)) / 255.0

    cam, _ = gradcam(input_tensor)
    seed = (cam > THRESHOLD).astype(np.float32)
    seg = refine_with_crf(image_np, seed)

    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.resize(gt_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    return compute_iou(seg, gt_bin)


def main() -> None:
    model = load_model('best_model_pretrained_flowers.pth')
    gradcampp = GradCAMPlusPlus(model, model.layer4[-1])

    ious: List[float] = []
    failures: List[int] = []
    count_75 = 0

    for i in tqdm(range(1, NUM_IMAGES + 1)):
        try:
            iou = process_image(i, gradcampp)
            if iou is None:
                failures.append(i)
                continue
            ious.append(iou)
            if iou >= 0.75:
                count_75 += 1
        except Exception as e:
            print(f"[!] Error at image {i}: {e}")
            failures.append(i)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    mean_iou = round(np.mean(ious), 4) if ious else "n/a"
    pct_75 = round(100 * count_75 / len(ious), 2) if ious else "n/a"

    df = pd.DataFrame([{
        "Mean IoU": mean_iou,
        "IoU ≥ 75%": pct_75,
        "Failed Images": len(failures)
    }])
    print(df)
    df.to_csv("rezultate_segmentare.csv", index=False)


if __name__ == "__main__":
    main()
