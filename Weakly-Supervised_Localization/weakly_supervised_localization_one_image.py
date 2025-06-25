import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
from typing import Tuple, List

# in this script, we will use a pre-trained VGG16 model to perform weakly-supervised localization
# for top-5 predictions on a single image from the VOC2012 dataset using Grad-CAM

class CamExtractor:
    def __init__(self, net: torch.nn.Module, layer: torch.nn.Module) -> None:
        self.net = net.eval()
        self.layer = layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register()

    def _register(self) -> None:
        def forward(m: torch.nn.Module, inp: torch.Tensor, out: torch.Tensor) -> None:
            self.activations = out.detach()

        def backward(m: torch.nn.Module, grad_in: Tuple[torch.Tensor], grad_out: Tuple[torch.Tensor]) -> None:
            self.gradients = grad_out[0].detach()

        self.layer.register_forward_hook(forward)
        self.layer.register_full_backward_hook(backward)

    def generate_map(self, x: torch.Tensor, label_id: int) -> np.ndarray:
        out = self.net(x)
        self.net.zero_grad()
        out[0, label_id].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np


def preprocess_img(img_path: str) -> Tuple[torch.Tensor, Image.Image]:
    proc = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return proc(img).unsqueeze(0), img


def overlay_boxes(
    original: Image.Image,
    preds: torch.Tensor,
    probs: torch.Tensor,
    extractor: CamExtractor,
    label_names: List[str]
) -> np.ndarray:
    img_draw = np.array(original.resize((224, 224))).copy()
    shades = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    for idx, (cid, conf) in enumerate(zip(preds, probs)):
        score_map = extractor.generate_map(input_tensor, cid.item())
        name = label_names[cid.item()]
        mask = (score_map > 0.5 * score_map.max()).astype(np.uint8) * 255
        n_l, labs, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if n_l > 1:
            biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            x, y, w, h = stats[biggest][:4]
            col = shades[idx % len(shades)]
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), col, 2)
            cv2.putText(img_draw, name, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            print(f"[{idx + 1}] {name:<25} | Confidence: {conf.item():.2f} | Box: ({x},{y}) → ({x+w},{y+h})")

    return img_draw

# I chose the image path to be a specific image from the VOC2012 dataset
image_path = "../VOC2012_train_val/JPEGImages/2007_004052.jpg"
input_tensor, original_image = preprocess_img(image_path)

vgg_weights = models.VGG16_Weights.DEFAULT
vgg_net = models.vgg16(weights=vgg_weights)
layer_to_watch = vgg_net.features[-1]
cam_tool = CamExtractor(vgg_net, layer_to_watch)

class_list = vgg_weights.meta["categories"]
model_output = vgg_net(input_tensor)
top5_probs, top5_ids = model_output[0].topk(5)

composite_image = overlay_boxes(original_image, top5_ids, top5_probs, cam_tool, class_list)

plt.imshow(composite_image)
plt.title("Grad-CAM Top-5 Heatmap Boxes")
plt.axis('off')
plt.show()
