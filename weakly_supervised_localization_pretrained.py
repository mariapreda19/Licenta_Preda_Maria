import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import xml.etree.ElementTree as ET

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
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def get_gt_boxes(annotation_path):
    boxes = {}
    root = ET.parse(annotation_path).getroot()
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        if name not in boxes:
            boxes[name] = []
        boxes[name].append((xmin, ymin, xmax, ymax))
    return boxes

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

selected_class = "person"
image_dir = r"C:\Users\40773\Desktop\Licenta\VOC2012_train_val\JPEGImages"
anno_dir = r"C:\Users\40773\Desktop\Licenta\VOC2012_train_val\Annotations"
test_list = r"C:\Users\40773\Desktop\Licenta\subset_test_images.txt"
num_images = 10

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

weights = models.VGG16_Weights.DEFAULT
model = models.vgg16(weights=weights)
grad_cam = GradCAM(model, model.features[28])
class_labels = weights.meta["categories"]
colors = [(255, 0, 0)]

with open(test_list, "r") as f:
    all_ids = [line.strip() for line in f.readlines()]

image_ids = []
for image_id in all_ids:
    xml_path = os.path.join(anno_dir, image_id + ".xml")
    if not os.path.exists(xml_path):
        continue
    gt_boxes = get_gt_boxes(xml_path)
    if selected_class in gt_boxes:
        image_ids.append(image_id)
    if len(image_ids) >= num_images:
        break

correct, total = 0, 0
visualizations = []

for image_id in tqdm(image_ids, desc="Processing"):
    img_path = os.path.join(image_dir, image_id + ".jpg")
    xml_path = os.path.join(anno_dir, image_id + ".xml")
    gt_boxes = get_gt_boxes(xml_path)
    image = Image.open(img_path).convert("RGB")
    w_orig, h_orig = image.size
    input_tensor = transform(image).unsqueeze(0)

    output = model(input_tensor)
    top5_prob, top5_cls = output[0].topk(5)

    image_resized = image.resize((224, 224))
    img_np = np.array(image_resized).copy()

    print(f"\nImage: {image_id}")
    for (xmin, ymin, xmax, ymax) in gt_boxes[selected_class]:
        print(f"   GT: {selected_class:<15} | Box: ({xmin}, {ymin}) – ({xmax}, {ymax})")
        x1 = int(xmin * 224 / w_orig)
        y1 = int(ymin * 224 / h_orig)
        x2 = int(xmax * 224 / w_orig)
        y2 = int(ymax * 224 / h_orig)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(img_np, selected_class, (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    match_found = False
    for i, (cls_id, prob) in enumerate(zip(top5_cls, top5_prob)):
        class_name = class_labels[cls_id.item()]
        if class_name != selected_class:
            continue

        heatmap = grad_cam.generate(input_tensor, cls_id.item())
        threshold = 0.3 * heatmap.max()
        binary_map = (heatmap > threshold).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

        if num_labels > 1:
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            x, y, w, h = stats[largest_idx][:4]
            x0 = int(x * w_orig / 224)
            y0 = int(y * h_orig / 224)
            x1 = int((x + w) * w_orig / 224)
            y1 = int((y + h) * h_orig / 224)
            pred_box = (x0, y0, x1, y1)

            for gt_box in gt_boxes[selected_class]:
                iou = compute_iou(pred_box, gt_box)
                if iou >= 0.5:
                    match_found = True
                    break

            cv2.rectangle(img_np, (x, y), (x + w, y + h), colors[0], 2)
            cv2.putText(img_np, f"{class_name} ({prob:.2f})", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 1)

        if match_found:
            break

    if match_found:
        correct += 1
    total += 1
    visualizations.append((img_np, image_id))

# ---------- Show Visualizations ----------
cols = 2
rows = (len(visualizations) + 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))

for i, (img, image_id) in enumerate(visualizations):
    ax = axes[i // cols, i % cols]
    ax.imshow(img)
    ax.set_title(image_id)
    ax.axis('off')

plt.tight_layout()
plt.show()

print(f"\n Localization Accuracy for class '{selected_class}' (Top-5 match, IoU ≥ 0.5):")
print(f"Correct: {correct} / {total} = {correct / total:.2%}")
