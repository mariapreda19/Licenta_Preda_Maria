import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image

# === Grad-CAM Utils ===
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        pooled_gradients = self.gradients.mean(dim=[0, 2, 3])
        activations = self.activations[0]
        weighted_activations = activations * pooled_gradients[:, None, None]
        heatmap = weighted_activations.sum(dim=0)
        heatmap = F.relu(heatmap).cpu().numpy()
        heatmap -= np.min(heatmap)
        heatmap /= (np.max(heatmap) + 1e-8)
        return heatmap

def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0), img

def overlay_heatmap(img_pil, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.4):
    img = np.array(img_pil)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed_img.astype(np.uint8)

def run_gradcam_on_predictions(model, target_layer, prediction_file, output_dir, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    os.makedirs(output_dir, exist_ok=True)
    with open(prediction_file, 'r') as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            path, true_class, pred_class, conf = line.strip().split('\t')
            input_tensor, original_img = preprocess_image(path)
            input_tensor = input_tensor.to(device)
            cam = GradCAM(model, target_layer)
            heatmap = cam.generate(input_tensor)
            result_img = overlay_heatmap(original_img, heatmap)

            subfolder = "correct" if true_class == pred_class else "wrong"
            save_dir = os.path.join(output_dir, subfolder)
            os.makedirs(save_dir, exist_ok=True)

            filename = os.path.basename(path).split('.')[0]
            out_path = os.path.join(save_dir, f"{filename}_{pred_class}_gradcam.png")
            cv2.imwrite(out_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            print(f"Saved Grad-CAM for {path} to {out_path}")

# === Example Usage (after model is trained and best_model.pt is saved) ===
class_names = ['beagle', 'golden_retriever', 'labrador_retriever', 'papillon', 'pug', 'siberian_husky']

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('best_model.pt'))
model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

run_gradcam_on_predictions(
    model=model,
    target_layer=model.layer4[1].conv2,  # This is the last conv layer of ResNet18
    prediction_file='predictions_on_test.txt',
    output_dir='cam_results',
    class_names=class_names
)
