# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import cv2
# import os
# from error_analysis import DogBreedCNN
#
#
# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model.eval()
#         self.target_layer = target_layer
#         self.gradients = None
#         self.activations = None
#         self._register_hooks()
#
#     def _register_hooks(self):
#         def forward_hook(module, input, output):
#             self.activations = output.detach()
#
#         def backward_hook(module, grad_input, grad_output):
#             self.gradients = grad_output[0].detach()
#
#         self.target_layer.register_forward_hook(forward_hook)
#         self.target_layer.register_full_backward_hook(backward_hook)
#
#     def generate(self, input_tensor, class_idx=None):
#         output = self.model(input_tensor)
#         if class_idx is None:
#             class_idx = output.argmax().item()
#
#         self.model.zero_grad()
#         output[0, class_idx].backward()
#
#         pooled_gradients = self.gradients.mean(dim=[0, 2, 3])
#         pooled_gradients = pooled_gradients.abs()
#
#         activations = self.activations[0]
#         weighted_activations = activations * pooled_gradients[:, None, None]
#
#         heatmap = weighted_activations.sum(dim=0)
#         heatmap = F.relu(heatmap).cpu().numpy()
#
#         heatmap -= np.min(heatmap)
#         heatmap /= (np.max(heatmap) + 1e-8)
#         return heatmap
#
#
# def preprocess_image(img_path):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])
#     img = Image.open(img_path).convert('RGB')
#     return transform(img).unsqueeze(0), img
#
#
# def overlay_heatmap(img_pil, heatmap, alpha=0.4):
#     img = np.array(img_pil)
#     heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap_uint8 = np.uint8(255 * heatmap_resized)
#     heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
#     heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
#     superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
#     return superimposed_img.astype(np.uint8)
#
#
# def run_gradcam_on_errors(predictions_file, model_weights, output_dir):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = DogBreedCNN().to(device)
#     model.load_state_dict(torch.load(model_weights, map_location=device))
#     target_layer = model.conv4[0]
#     cam = GradCAM(model, target_layer)
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     with open(predictions_file, 'r') as f:
#         lines = f.readlines()[1:]
#
#     for line in lines:
#         path, true_cls, pred_cls, conf = line.strip().split('\t')
#         if true_cls != pred_cls:
#             input_tensor, original_img = preprocess_image(path)
#             input_tensor = input_tensor.to(device)
#             heatmap = cam.generate(input_tensor)
#             result_img = overlay_heatmap(original_img, heatmap)
#             filename = os.path.basename(path)
#             save_path = os.path.join(output_dir, f"{true_cls}_as_{pred_cls}_{filename}")
#             cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
#             print(f"Saved: {save_path}")
#
#
# if __name__ == '__main__':
#     run_gradcam_on_errors(
#         predictions_file='predictions_on_test.txt',
#         model_weights='best_model.pt',
#         output_dir='cam_errors'
#     )


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import VGG16_Weights
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import urllib



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
        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()

        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Gradient pooling
        pooled_gradients = self.gradients.mean(dim=[0, 2, 3])
        # Optional: use abs to capture both positive and negative importance
        pooled_gradients = pooled_gradients.abs()

        # Activation weighting
        activations = self.activations[0]
        weighted_activations = activations * pooled_gradients[:, None, None]

        # Combine and apply ReLU
        heatmap = weighted_activations.sum(dim=0)
        heatmap = F.relu(heatmap).cpu().numpy()

        # Normalize (no inversion)
        heatmap -= np.min(heatmap)
        heatmap /= (np.max(heatmap) + 1e-8)

        return heatmap, class_idx


def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0), img


def overlay_heatmap(img_pil, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.4, threshold=0.0):
    img = np.array(img_pil)

    # Resize the heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Normalize the heatmap again for safety
    heatmap_resized -= np.min(heatmap_resized)
    heatmap_resized /= (np.max(heatmap_resized) + 1e-8)

    # Apply threshold to suppress weak activations (optional)
    heatmap_resized = np.where(heatmap_resized > threshold, heatmap_resized, 0)

    # Convert to 8-bit and apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  # Convert to RGB for PIL compatibility

    # Combine the heatmap with the image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed_img.astype(np.uint8)




def replace_relu_with_clone(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            replace_relu_with_clone(child)


def run_grad_cam_on_image(img_path, class_idx_override=None, save_path='gradcam_output.jpg'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.vgg16(weights=VGG16_Weights.DEFAULT).to(device)
    replace_relu_with_clone(model)

    target_layer = model.features[29]
    cam = GradCAM(model, target_layer)

    input_tensor, original_img = preprocess_image(img_path)
    input_tensor = input_tensor.to(device)

    output = model(input_tensor)
    top5 = torch.topk(output, 5)

    LABELS_URL = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    classes = urllib.request.urlopen(LABELS_URL).read().decode('utf-8').splitlines()

    print("\nTop 5 predictions:")
    for rank, idx in enumerate(top5.indices[0]):
        print(f"{rank + 1}. {classes[idx]} (index: {idx.item()}) - score: {output[0, idx].item():.4f}")

    class_idx = class_idx_override if class_idx_override is not None else output.argmax().item()
    heatmap, predicted_class_idx = cam.generate(input_tensor, class_idx=class_idx)
    plt.figure(figsize=(6, 4))
    plt.imshow(heatmap, cmap='jet', interpolation='bicubic')
    plt.colorbar()
    plt.title("Raw Normalized Heatmap")
    plt.show()

    result_img = overlay_heatmap(original_img, heatmap)

    cv2.imwrite(save_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print(f"\nVisualized class: {classes[class_idx]}")
    print(f"Saved Grad-CAM result to {save_path}")

    # Show side-by-side: raw heatmap + overlayed image
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(heatmap, cmap='jet', interpolation='bicubic')
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(result_img)
    plt.title(f"Predicted: {classes[class_idx]}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    image_path = 'bias_2/train/constructor/consta9.jpg'
    run_grad_cam_on_image(image_path)

# from PIL import Image
#
# # Path to the input image
# input_path = "datasets-Error_Analysis/Test/golden_retirever/n02099601_2440.jpg"
#
# # Open the image
# image = Image.open(input_path)
#
# # Resize to 100x100 pixels
# resized_image = image.resize((100, 100))
#
# # Save the resized image
# resized_image.save("test_images/chihuahua")
#
# # Optionally, show it
# resized_image.show()
