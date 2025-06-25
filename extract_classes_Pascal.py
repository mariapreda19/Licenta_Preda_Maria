import os
import xml.etree.ElementTree as ET
from collections import defaultdict
import random


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]
VOC_SET = set(VOC_CLASSES)

annotations_dir = r"C:\Users\40773\Desktop\Licenta\VOC2012_train_val\Annotations"
MAX_TRAIN = 100
MAX_TEST = 25


img_to_labels = {}
label_to_imgs = defaultdict(list)

for fname in os.listdir(annotations_dir):
    if not fname.endswith(".xml"):
        continue
    img_id = fname.replace(".xml", "")
    tree = ET.parse(os.path.join(annotations_dir, fname))
    root = tree.getroot()
    labels = set()
    for obj in root.findall("object"):
        cls = obj.find("name").text.strip()
        if cls in VOC_SET:
            labels.add(cls)
    if labels:
        img_to_labels[img_id] = labels
        for cls in labels:
            label_to_imgs[cls].append(img_id)


train_ids, test_ids = set(), set()
used_ids = set()

class_train_counts = defaultdict(int)
class_test_counts = defaultdict(int)

random.seed(42)

for cls in VOC_CLASSES:
    candidates = [img for img in label_to_imgs[cls] if img not in used_ids]
    random.shuffle(candidates)

    train_count = 0
    test_count = 0
    for img in candidates:
        if train_count < MAX_TRAIN and img not in train_ids:
            train_ids.add(img)
            used_ids.add(img)
            train_count += 1
        elif test_count < MAX_TEST and img not in test_ids and img not in train_ids:
            test_ids.add(img)
            used_ids.add(img)
            test_count += 1
        if train_count >= MAX_TRAIN and test_count >= MAX_TEST:
            break
    print(f"[{cls:<12}] Train: {train_count} | Test: {test_count}")


def write_labels(file_txt, file_csv, img_ids):
    with open(file_txt, "w") as f:
        for img_id in sorted(img_ids):
            f.write(img_id + "\n")

    with open(file_csv, "w") as f:
        f.write("image_id,labels\n")
        for img_id in sorted(img_ids):
            labels = ",".join(sorted(img_to_labels[img_id]))
            f.write(f"{img_id},{labels}\n")

write_labels("subset_train_images.txt", "subset_train_labels.csv", train_ids)
write_labels("subset_test_images.txt", "subset_test_labels.csv", test_ids)

print(f"\n Final counts:")
print(f"Train: {len(train_ids)} images")
print(f"Test: {len(test_ids)} images")
