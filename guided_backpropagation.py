import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import VGG16_Weights
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import urllib


class GuidedBackpropReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        ctx.save_for_backward(positive_mask)
        return input * positive_mask

    @staticmethod
    def backward(ctx, grad_output):
        positive_mask, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[grad_output < 0] = 0
        return grad_input * positive_mask


class GuidedReLU(nn.Module):
    def forward(self, input):
        return GuidedBackpropReLUFunction.apply(input)


def replace_relu_with_guided_relu(model):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, GuidedReLU())
        else:
            replace_relu_with_guided_relu(module)


def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0), img


def guided_backprop(model, input_tensor, class_idx):
    model.zero_grad()
    output = model(input_tensor)
    target = output[0, class_idx]
    target.backward()

    grad = input_tensor.grad[0].cpu().numpy()
    grad = np.transpose(grad, (1, 2, 0))  # CHW → HWC
    grad -= grad.min()
    grad /= grad.max() + 1e-8
    return grad


def run_guided_backprop(img_path, class_idx_override=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(weights=VGG16_Weights.DEFAULT).to(device)
    replace_relu_with_guided_relu(model)
    model.eval()

    input_tensor, original_img = preprocess_image(img_path)
    input_tensor = input_tensor.to(device).requires_grad_()

    # Prediction and class selection
    output = model(input_tensor)
    class_idx = class_idx_override if class_idx_override is not None else output.argmax().item()

    # Download labels
    LABELS_URL = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    classes = urllib.request.urlopen(LABELS_URL).read().decode('utf-8').splitlines()
    print(f"\nPredicted class: {classes[class_idx]} (index {class_idx})")

    # Guided Backpropagation
    guided_grad = guided_backprop(model, input_tensor, class_idx)

    # Show result
    plt.figure(figsize=(6, 6))
    plt.imshow(guided_grad)
    plt.title(f"Guided Backpropagation - {classes[class_idx]}")
    plt.axis('off')
    plt.show()

    # Save result
    output_path = "guided_backprop_output.jpg"
    cv2.imwrite(output_path, np.uint8(255 * guided_grad[..., ::-1]))  # Convert RGB to BGR
    print(f"Saved Guided Backpropagation result to {output_path}")


if __name__ == '__main__':
    run_guided_backprop("datasets-Error_Analysis/Test/golden_retirever/n02099601_2440.jpg")
