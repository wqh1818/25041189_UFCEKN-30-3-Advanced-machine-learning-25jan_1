--------------------------------------------------------------
# Setup
----------------------------------------------------------------

# Install dependencies
!pip install -q opencv-python-headless scipy matplotlib seaborn pandas Pillow
# For HuggingFace dataset access
!pip install -q datasets

import os, json, xml.etree.ElementTree as ET, urllib.request, zipfile, shutil, random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
import cv2
from scipy.ndimage import label as scipy_label

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as tv_models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
torch.manual_seed(42)
np.random.seed(42)



--------------------------------------------------------------
# VGG-16 from Scratch
----------------------------------------------------------------

class VGG16(nn.Module):
    """
    VGG-16 implemented from scratch following Simonyan & Zisserman (2014).

    Architecture (configuration D):
      Block 1: conv3-64, conv3-64, maxpool
      Block 2: conv3-128, conv3-128, maxpool
      Block 3: conv3-256, conv3-256, conv3-256, maxpool
      Block 4: conv3-512, conv3-512, conv3-512, maxpool
      Block 5: conv3-512, conv3-512, conv3-512, maxpool
      FC: 4096 -> 4096 -> 1000

    The final conv layer (block5, layer3) is used as the target layer
    for Grad-CAM, producing 14x14 feature maps for a 224x224 input.
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()

        # Convolutional feature extractor
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.features = nn.Sequential(
            self.block1, self.block2, self.block3,
            self.block4, self.block5
        )

        # Adaptive pool so we can handle non-224 inputs
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Fully-connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self) -> nn.Module:
        # Return the last conv layer (used as Grad-CAM hook target).
        return self.block5[-3]


# Verify architecture matches torchvision exactly
model_scratch = VGG16(num_classes=1000)
ref = tv_models.vgg16()

scratch_params = sum(p.numel() for p in model_scratch.parameters())
ref_params = sum(p.numel() for p in ref.parameters())

print(f'Our VGG-16 : {scratch_params:,} parameters')
print(f'torchvision : {ref_params:,} parameters')
assert scratch_params == ref_params, 'Parameter count mismatch — check architecture!'
print('Architecture verified ✓')




--------------------------------------------------------------
# Load Pre-trained Weights
----------------------------------------------------------------

def load_pretrained_weights(model: VGG16) -> VGG16:
    ref = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
    ref_sd = ref.state_dict()
    our_sd = model.state_dict()

    our_keys_filtered = [k for k in our_sd.keys() if k.startswith('block') or k.startswith('classifier')]
    ref_keys = list(ref_sd.keys())

    assert len(our_keys_filtered) == len(ref_keys), \
        f'Still mismatched: ours={len(our_keys_filtered)}, ref={len(ref_keys)}'

    new_sd = dict(our_sd)
    for our_key, ref_key in zip(our_keys_filtered, ref_keys):
        assert our_sd[our_key].shape == ref_sd[ref_key].shape, \
            f'Shape mismatch: {our_key} {our_sd[our_key].shape} vs {ref_key} {ref_sd[ref_key].shape}'
        new_sd[our_key] = ref_sd[ref_key]

    model.load_state_dict(new_sd)
    print('Pre-trained weights loaded successfully ✓')
    return model


vgg16 = VGG16(num_classes=1000).to(DEVICE)
vgg16 = load_pretrained_weights(vgg16)
vgg16.eval()

with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224).to(DEVICE)
    out = vgg16(dummy)
    print(f'Output shape: {out.shape}  (expected [1, 1000])')



--------------------------------------------------------------
# Grad-CAM Implementation
----------------------------------------------------------------

class GradCAM:
    """
    Grad-CAM implementation using PyTorch forward/backward hooks.
    Follows Equation (1)-(2) from Selvaraju et al. (2019):
      alpha_k^c = (1/Z) * sum_{i,j} (d y^c / d A^k_{ij})   [Eq. 1]
      L^c = ReLU(sum_k alpha_k^c * A^k)                     [Eq. 2]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self._activations = output.detach().clone()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach().clone()

        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self,
                 image_tensor: torch.Tensor,
                 class_idx: int = None) -> tuple:

        # Generate a Grad-CAM heatmap.
        self.model.zero_grad()
        image_tensor = image_tensor.clone().requires_grad_(True)

        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backprop only through the target class score
        score = logits[0, class_idx]
        score.backward()

        # alpha_k^c = GAP of gradients  [Eq. 1]
        gradients = self._gradients
        activations = self._activations
        alpha = gradients.mean(dim=(2, 3), keepdim=True)
        # Weighted combination + ReLU  [Eq. 2]
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to input spatial dimensions
        H, W = image_tensor.shape[2], image_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_idx, logits.detach()



--------------------------------------------------------------
# Modified CAM for VGG-16
----------------------------------------------------------------

class VGG16_CAM(nn.Module):
    """
    VGG-16 modified for standard CAM compatibility.
    Replaces the three FC layers with GlobalAveragePooling + Linear(512, num_classes).
    The conv backbone is shared/transferred from the standard VGG-16.
    """

    def __init__(self, num_classes: int = 1000):
        super().__init__()

        # Same conv blocks as VGG16
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2))
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=False),
            # NO MaxPool here to keep spatial resolution for CAM
        )

        # CAM head: GAP + single linear layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_cam_weights(self, class_idx: int) -> torch.Tensor:
        """Return the linear weights for a given class — used to compute CAM."""
        return self.classifier.weight[class_idx]

    def get_last_conv_layer(self) -> nn.Module:
        return self.block5[-2]


def transfer_conv_weights(src: VGG16, dst: VGG16_CAM) -> VGG16_CAM:
    """
    Copy the 13 shared conv layers from the standard VGG-16 into VGG16_CAM.
    Only the classifier head of dst remains randomly initialised (to be fine-tuned).
    """
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()

    conv_keys_src = [k for k in src_sd if 'block' in k]
    conv_keys_dst = [k for k in dst_sd if 'block' in k]

    assert len(conv_keys_src) == len(conv_keys_dst)

    for ks, kd in zip(conv_keys_src, conv_keys_dst):
        assert src_sd[ks].shape == dst_sd[kd].shape
        dst_sd[kd] = src_sd[ks]

    dst.load_state_dict(dst_sd)
    print(f'Transferred {len(conv_keys_src)} conv parameter tensors ✓')
    return dst


vgg16_cam = VGG16_CAM(num_classes=1000).to(DEVICE)
vgg16_cam = transfer_conv_weights(vgg16, vgg16_cam)


class CAM:
    """
    CAM formula:
      M^c(x, y) = sum_k  w_k^c * f_k(x, y)
    where w_k^c are the linear classifier weights for class c,
    and f_k are the spatial activations of the last conv feature map k.

    This is only valid because we have GAP → Linear architecture.
    """

    def __init__(self, model: VGG16_CAM):
        self.model = model
        self._features = None
        self._register_hook()

    def _register_hook(self):
        def hook(module, input, output):
            self._features = output.detach()
        self._handle = self.model.get_last_conv_layer().register_forward_hook(hook)

    def remove_hook(self):
        self._handle.remove()

    def generate(self,
                 image_tensor: torch.Tensor,
                 class_idx: int = None) -> tuple:

        # Generate a CAM heatmap.
        with torch.no_grad():
            logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Get classifier weights for the target class
        weights = self.model.get_cam_weights(class_idx)

        # Weighted sum of feature maps
        features = self._features.squeeze(0)
        cam = torch.einsum('k,khw->hw', weights, features)
        cam = F.relu(cam)

        # Resize to input spatial dimensions
        H, W = image_tensor.shape[2], image_tensor.shape[3]
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()

        # Normalise to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam, class_idx, logits.detach()



--------------------------------------------------------------
# Datasets and Fine-tune VGG16-CAM Classifier Head
----------------------------------------------------------------

# ImageNet preprocessing
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])



import torchvision.datasets as dsets
from torchvision.datasets import OxfordIIITPet
import xml.etree.ElementTree as ET
from pathlib import Path

DATA_DIR = '/content/oxford_pets'

# Load images and category labels only
pet_dataset = OxfordIIITPet(
    root = DATA_DIR,
    split = 'trainval',
    target_types = ['category'],
    download = True,
    transform = None
)

print(f'Dataset size: {len(pet_dataset)} images')
print(f'Number of classes: {len(pet_dataset.classes)}')

def get_bbox_from_xml(xml_path: str) -> list:
    """
    Parse the Oxford Pet XML annotation file to extract bounding boxes.
    Returns list of [xmin, ymin, xmax, ymax] in pixel coordinates.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                bbox = [
                    int(float(bndbox.find('xmin').text)),
                    int(float(bndbox.find('ymin').text)),
                    int(float(bndbox.find('xmax').text)),
                    int(float(bndbox.find('ymax').text)),
                ]
                bboxes.append(bbox)
        return bboxes
    except Exception:
        return []

# Build a mapping from image filename stem -> bbox
XML_DIR = Path('/content/oxford_pets/oxford-iiit-pet/annotations/xmls')
filename_to_bbox = {}
for xml_file in XML_DIR.glob('*.xml'):
    bboxes = get_bbox_from_xml(str(xml_file))
    if bboxes:
        filename_to_bbox[xml_file.stem] = bboxes

print(f'Loaded bboxes for {len(filename_to_bbox)} images')
print(f'Example keys: {list(filename_to_bbox.keys())[:3]}')

# Check what attribute holds image paths in this torchvision version
img_attr = None
for attr in ['images', '_images', 'imgs', 'samples']:
    if hasattr(pet_dataset, attr):
        img_attr = attr
        print(f'Image path attribute: .{attr}')
        val = getattr(pet_dataset, attr)
        print(f'First entry: {val[0]}')
        break

# Verify one sample
sample_img, sample_label = pet_dataset[0]
sample_name = Path(pet_dataset._images[0]).stem
print(f'\nSample image name : {sample_name}')
print(f'Sample label : {sample_label} ({pet_dataset.classes[sample_label]})')
print(f'Sample bbox : {filename_to_bbox.get(sample_name, "NOT FOUND")}')
print(f'Image size : {sample_img.size}')




PET_CLASS_TO_IMAGENET = {
    'Abyssinian': 285,
    'Bengal': 281,
    'Birman': 284,
    'Bombay': 282,
    'British_Shorthair': 283,
    'Maine_Coon': 284,
    'Persian': 283,
    'Ragdoll': 281,
    'Russian_Blue': 281,
    'Siamese': 284,
    'american_bulldog': 149,
    'american_pit_bull_terrier': 232,
    'basset_hound': 162,
    'beagle': 162,
    'boxer': 242,
    'chihuahua': 151,
    'english_cocker_spaniel': 219,
    'english_setter': 206,
    'great_pyrenees': 257,
    'pug': 254,
}

EVAL_WNIDS = list(PET_CLASS_TO_IMAGENET.keys())[:10]
EVAL_CLASS_INDICES = [PET_CLASS_TO_IMAGENET[c] for c in EVAL_WNIDS]

pet_localidx_to_imagenet = {}
for local_idx, cls_name in enumerate(pet_dataset.classes):
    if cls_name in PET_CLASS_TO_IMAGENET:
        pet_localidx_to_imagenet[local_idx] = PET_CLASS_TO_IMAGENET[cls_name]

print(f'Using {len(EVAL_WNIDS)} classes: {EVAL_WNIDS}')
print(f'ImageNet indices: {EVAL_CLASS_INDICES}')



from collections import defaultdict

def collect_pet_samples(dataset, pet_localidx_to_imagenet,
                        filename_to_bbox, max_per_class=50):
    counts = defaultdict(int)
    samples = []
    target_set = set(pet_localidx_to_imagenet.keys())
    target_imagenet = set(PET_CLASS_TO_IMAGENET[c] for c in EVAL_WNIDS)

    # Find the attribute that stores image file paths
    img_paths = None
    for attr in ['_images', '_images', 'imgs', 'samples']:
        if hasattr(dataset, attr):
            img_paths = getattr(dataset, attr)
            break

    if img_paths is None:
        raise RuntimeError('Cannot find image path attribute on dataset')

    for idx in range(len(dataset)):
        img, local_label = dataset[idx]

        if local_label not in target_set:
            continue

        imagenet_label = pet_localidx_to_imagenet[local_label]
        if imagenet_label not in target_imagenet:
            continue
        if counts[imagenet_label] >= max_per_class:
            continue

        # Get the image filename stem
        img_path = img_paths[idx]
        if isinstance(img_path, (list, tuple)):
            img_path = img_path[0]
        img_stem = Path(str(img_path)).stem

        bboxes = filename_to_bbox.get(img_stem, [])

        samples.append({
            'image' : img,
            'label' : imagenet_label,
            'bbox' : bboxes,
        })
        counts[imagenet_label] += 1

    n_with_bbox = sum(1 for s in samples if s['bbox'])
    print(f'Collected {len(samples)} samples')
    print(f'Samples with bboxes: {n_with_bbox}/{len(samples)}')
    if samples:
        s = samples[0]
        print(f'\nFirst sample: label={s["label"]}, bbox={s["bbox"]}')
    return samples


all_samples = collect_pet_samples(
    pet_dataset, pet_localidx_to_imagenet, filename_to_bbox, max_per_class=50)



from torch.utils.data import Dataset, DataLoader
import random

class PetDataset(Dataset):
    def __init__(self, samples, transform):
        unique = sorted(set(s['label'] for s in samples))
        self.label_map = {v: i for i, v in enumerate(unique)}
        self.inv_label_map = {i: v for v, i in self.label_map.items()}
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = s['image'].convert('RGB')
        label = self.label_map[s['label']]
        return self.transform(image), label


random.shuffle(all_samples)
split = int(0.8 * len(all_samples))
train_data = PetDataset(all_samples[:split], transform=train_transform)
val_data = PetDataset(all_samples[split:], transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,  num_workers=2)
val_loader = DataLoader(val_data,   batch_size=32, shuffle=False, num_workers=2)

NUM_CLASSES = len(train_data.label_map)

# Store the inverse map globally so evaluator can use it
CAM_INV_LABEL_MAP = train_data.inv_label_map
print(f'Train: {len(train_data)} | Val: {len(val_data)} | Classes: {NUM_CLASSES}')
print(f'Local->ImageNet mapping: {CAM_INV_LABEL_MAP}')

# Fine-tune only the CAM head
vgg16_cam.classifier = nn.Linear(512, NUM_CLASSES).to(DEVICE)
for name, param in vgg16_cam.named_parameters():
    param.requires_grad = ('classifier' in name)

optimizer = torch.optim.Adam(vgg16_cam.classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
cam_history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

for epoch in range(5):
    vgg16_cam.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(vgg16_cam(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    vgg16_cam.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = vgg16_cam(images)
            val_loss += criterion(out, labels).item() * images.size(0)
            correct += (out.argmax(1) == labels).sum().item()

    tl = running_loss / len(train_data)
    vl = val_loss / len(val_data)
    va = correct / len(val_data)
    cam_history['train_loss'].append(tl)
    cam_history['val_loss'].append(vl)
    cam_history['val_acc'].append(va)
    print(f'Epoch {epoch+1}/5  Train Loss: {tl:.4f}  Val Loss: {vl:.4f}  Val Acc: {va:.3f}')



--------------------------------------------------------------
# Bounding Box Localisation Utilities
----------------------------------------------------------------


def heatmap_to_bbox(heatmap: np.ndarray, threshold: float = 0.15) -> tuple:
    """
    Convert a normalised heatmap [H, W] into a bounding box.
    Following the paper: threshold at 15% of max, take largest connected segment.

    Returns:
        (x1, y1, x2, y2) in pixel coordinates, or None if no region found.
    """
    binary = (heatmap >= threshold).astype(np.uint8)

    if binary.sum() == 0:
        return None

    # Find connected components
    labeled, num_features = scipy_label(binary)
    if num_features == 0:
        return None

    # Take the largest component
    sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
    largest_id = np.argmax(sizes) + 1
    component = (labeled == largest_id)

    rows = np.where(component.any(axis=1))[0]
    cols = np.where(component.any(axis=0))[0]

    x1, y1 = int(cols.min()), int(rows.min())
    x2, y2 = int(cols.max()), int(rows.max())

    return (x1, y1, x2, y2)


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    # Compute Intersection over Union between two boxes.

    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def scale_bbox_to_image(bbox: tuple, from_size: tuple, to_size: tuple) -> tuple:
    """
    Rescale a bounding box from one image size to another.
    from_size, to_size: (width, height)
    """
    sx = to_size[0] / from_size[0]
    sy = to_size[1] / from_size[1]
    x1, y1, x2, y2 = bbox
    return (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))


def is_correctly_localised(pred_bbox, gt_bboxes, iou_threshold: float = 0.5) -> bool:
    """
    Returns True if the predicted bbox has IoU >= threshold with ANY gt box.
    (ImageNet often has multiple ground-truth boxes per image.)
    """
    if pred_bbox is None or not gt_bboxes:
        return False
    return any(compute_iou(pred_bbox, gt) >= iou_threshold for gt in gt_bboxes)



--------------------------------------------------------------
# Quantitative Localization Evaluation
----------------------------------------------------------------

def evaluate_localization(cam_method,
    eval_samples: list,
    orig_class_indices: list,
    method_name: str = 'Grad-CAM',
    iou_threshold: float = 0.5,
    heatmap_threshold: float = 0.15) -> dict:

    records = []
    n_top1_cls_correct = 0
    n_top5_cls_correct = 0
    n_top1_loc_correct = 0
    n_top5_loc_correct = 0

    for i, sample in enumerate(eval_samples):
        image = sample['image'].convert('RGB')
        gt_label = sample['label']
        gt_bboxes = sample['bbox']

        orig_w, orig_h = image.size

        # Preprocess
        tensor = val_transform(image).unsqueeze(0).to(DEVICE)

        # Generate heatmap (use top-1 predicted class)
        heatmap, pred_class_local, logits = cam_method.generate(tensor, class_idx=None)

        # Map prediction back to original ImageNet label space
        topk5_local = logits[0].topk(5).indices.cpu().tolist()
        top1_local = topk5_local[0]

        # For CAM map local->original; for Grad-CAM already 1000-class
        if hasattr(cam_method, '_is_local') and cam_method._is_local:
            # Use the stored inv_label_map so local 0..N maps correctly to ImageNet indices
            inv_map = cam_method._inv_label_map
            top1_orig = inv_map.get(top1_local, -1)
            topk5_orig = [inv_map.get(j, -1) for j in topk5_local]
        else:
            top1_orig = top1_local
            topk5_orig = topk5_local

        cls_correct_top1 = (top1_orig == gt_label)
        cls_correct_top5 = (gt_label in topk5_orig)

        n_top1_cls_correct += int(cls_correct_top1)
        n_top5_cls_correct += int(cls_correct_top5)

        # Generate predicted bounding box from heatmap
        pred_bbox_224 = heatmap_to_bbox(heatmap, threshold=heatmap_threshold)

        # Scale predicted bbox to original image dimensions
        if pred_bbox_224 is not None:
            pred_bbox_orig = scale_bbox_to_image(pred_bbox_224, (224, 224), (orig_w, orig_h))
        else:
            pred_bbox_orig = None

        # Localization: class must be correct AND bbox IoU >= threshold
        loc_top1 = cls_correct_top1 and is_correctly_localised(pred_bbox_orig, gt_bboxes, iou_threshold)
        loc_top5 = cls_correct_top5 and is_correctly_localised(pred_bbox_orig, gt_bboxes, iou_threshold)

        n_top1_loc_correct += int(loc_top1)
        n_top5_loc_correct += int(loc_top5)

        records.append({
            'image_idx' : i,
            'gt_label' : gt_label,
            'top1_pred' : top1_orig,
            'cls_top1' : cls_correct_top1,
            'cls_top5' : cls_correct_top5,
            'loc_top1' : loc_top1,
            'loc_top5' : loc_top5,
            'heatmap' : heatmap,
            'pred_bbox' : pred_bbox_orig,
            'gt_bboxes' : gt_bboxes,
            'image' : image,
        })

        if (i + 1) % 50 == 0:
            print(f'  [{method_name}] Processed {i+1}/{len(eval_samples)}')

    N = len(eval_samples)
    results = {
        'method' : method_name,
        'N' : N,
        'cls_top1_acc' : n_top1_cls_correct / N * 100,
        'cls_top5_acc' : n_top5_cls_correct / N * 100,
        'loc_top1_acc' : n_top1_loc_correct / N * 100,
        'loc_top5_acc' : n_top5_loc_correct / N * 100,
        'loc_top1_err' : (1 - n_top1_loc_correct / N) * 100,
        'loc_top5_err' : (1 - n_top5_loc_correct / N) * 100,
        'records' : records,
    }
    return results


print('Evaluation function ready.')



vgg16.eval()
vgg16_cam.eval()

grad_cam_method = GradCAM(model=vgg16, target_layer=vgg16.get_last_conv_layer())
cam_method = CAM(model=vgg16_cam)
cam_method._is_local = True
cam_method._inv_label_map = CAM_INV_LABEL_MAP

print('Running Grad-CAM evaluation...')
gradcam_results = evaluate_localization(
    grad_cam_method, all_samples,
    orig_class_indices=EVAL_CLASS_INDICES,
    method_name='Grad-CAM')

print('\nRunning CAM evaluation...')
cam_results = evaluate_localization(
    cam_method, all_samples,
    orig_class_indices=EVAL_CLASS_INDICES,
    method_name='CAM (Modified)')

grad_cam_method.remove_hooks()
cam_method.remove_hook()



--------------------------------------------------------------
# Result table
----------------------------------------------------------------

# Build comparison table
rows = []
for res in [gradcam_results, cam_results]:
    rows.append({
        'Method': res['method'],
        'N (images)': res['N'],
        'Cls Top-1 Acc (%)': f"{res['cls_top1_acc']:.2f}",
        'Cls Top-5 Acc (%)': f"{res['cls_top5_acc']:.2f}",
        'Loc Top-1 Err (%)': f"{res['loc_top1_err']:.2f}",
        'Loc Top-5 Err (%)': f"{res['loc_top5_err']:.2f}",
    })

# Paper's reported numbers for reference (VGG-16 on full ILSVRC-15)
rows.append({
    'Method': 'Grad-CAM [Paper, VGG-16, ILSVRC-15]',
    'N (images)': '50,000',
    'Cls Top-1 Acc (%)' : '69.62',
    'Cls Top-5 Acc (%)' : '89.11',
    'Loc Top-1 Err (%)' : '56.51',
    'Loc Top-5 Err (%)' : '46.41',
})
rows.append({
    'Method' : 'CAM [Paper, VGG-16-GAP, ILSVRC-15]',
    'N (images)' : '50,000',
    'Cls Top-1 Acc (%)' : '67.09',
    'Cls Top-5 Acc (%)' : '88.11',
    'Loc Top-1 Err (%)' : '57.20',
    'Loc Top-5 Err (%)' : '45.14',
})

df = pd.DataFrame(rows)
print('=' * 90)
print('Table 1 — Classification and Localisation Performance on ImageNet Subset')
print('(Lower localization error = better. Paper results on full ILSVRC-15 val for reference.)')
print('=' * 90)
print(df.to_string(index=False))
print('=' * 90)



--------------------------------------------------------------
# Visual Comparison
----------------------------------------------------------------

def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5):

    img_array = np.array(image.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_color = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_color + (1 - alpha) * img_array).astype(np.uint8)
    return overlay


def draw_bboxes_on_ax(ax, pred_bbox, gt_bboxes, img_size=(224, 224)):
    # Draw predicted (red) and ground-truth (green) bboxes on a matplotlib axis.
    W, H = img_size
    if pred_bbox is not None:
        x1, y1, x2, y2 = scale_bbox_to_image(pred_bbox, (W, H), (224, 224))
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor='red', facecolor='none',
                                  label='Predicted')
        ax.add_patch(rect)

    for gt in gt_bboxes:
        gx1, gy1, gx2, gy2 = scale_bbox_to_image(gt, (W, H), (224, 224))
        rect = patches.Rectangle((gx1, gy1), gx2-gx1, gy2-gy1,
                                  linewidth=2, edgecolor='lime', facecolor='none',
                                  linestyle='--', label='Ground Truth')
        ax.add_patch(rect)


# Select 6 diverse samples for visualisation
# Pick 1 per class
viz_indices = []
seen_labels = set()
for r in gradcam_results['records']:
    if r['gt_label'] not in seen_labels and r['cls_top1']:
        viz_indices.append(r['image_idx'])
        seen_labels.add(r['gt_label'])
    if len(viz_indices) == 6:
        break

fig, axes = plt.subplots(len(viz_indices), 4, figsize=(18, 4 * len(viz_indices)))
col_titles = ['Original Image', 'Grad-CAM Heatmap', 'CAM Heatmap', 'Side-by-Side']

for row, idx in enumerate(viz_indices):
    gc_rec = gradcam_results['records'][idx]
    cam_rec = cam_results['records'][idx]
    image = gc_rec['image']
    orig_w, orig_h = image.size

    # Column 0: Original image with GT bbox
    ax = axes[row, 0]
    ax.imshow(image.resize((224, 224)))
    draw_bboxes_on_ax(ax, None, gc_rec['gt_bboxes'], (orig_w, orig_h))
    ax.set_title(f'Class: {gc_rec["gt_label"]}', fontsize=9)
    ax.axis('off')

    # Column 1: Grad-CAM
    ax = axes[row, 1]
    gc_overlay = overlay_heatmap(image, gc_rec['heatmap'])
    ax.imshow(gc_overlay)
    draw_bboxes_on_ax(ax, gc_rec['pred_bbox'], gc_rec['gt_bboxes'], (orig_w, orig_h))
    loc_str = '✓' if gc_rec['loc_top1'] else '✗'
    ax.set_title(f'Grad-CAM  {loc_str}', fontsize=9)
    ax.axis('off')

    # Column 2: CAM
    ax = axes[row, 2]
    cam_overlay = overlay_heatmap(image, cam_rec['heatmap'])
    ax.imshow(cam_overlay)
    draw_bboxes_on_ax(ax, cam_rec['pred_bbox'], cam_rec['gt_bboxes'], (orig_w, orig_h))
    loc_str = '✓' if cam_rec['loc_top1'] else '✗'
    ax.set_title(f'CAM (Modified)  {loc_str}', fontsize=9)
    ax.axis('off')

    # Column 3: Heatmap difference
    ax = axes[row, 3]
    diff = gc_rec['heatmap'] - cam_rec['heatmap']
    im = ax.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.6)
    ax.set_title('Grad-CAM − CAM\n(blue=CAM stronger, red=Grad-CAM stronger)', fontsize=8)
    ax.axis('off')

# Column headers
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=11, fontweight='bold', pad=12)

plt.suptitle('Grad-CAM vs Modified CAM — Qualitative Comparison\n'
             'Green dashed = Ground Truth, Red solid = Predicted Bbox',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('gradcam_vs_cam_qualitative.png', dpi=150, bbox_inches='tight')
plt.show()



--------------------------------------------------------------
# Per-Class Localization Breakdown
----------------------------------------------------------------

def per_class_breakdown(results: dict, wnids: list, class_indices: list) -> pd.DataFrame:
    # Compute per-class localization accuracy.
    class_records = defaultdict(list)
    for r in results['records']:
        class_records[r['gt_label']].append(r)

    rows = []
    for wnid, cidx in zip(wnids, class_indices):
        recs = class_records.get(cidx, [])
        if not recs:
            continue
        n = len(recs)
        rows.append({
            'WNID': wnid,
            'N' : n,
            'Cls Acc (%)' : sum(r['cls_top1'] for r in recs) / n * 100,
            'Loc Top-1 Acc (%)' : sum(r['loc_top1'] for r in recs) / n * 100,
            'Loc Top-5 Acc (%)' : sum(r['loc_top5'] for r in recs) / n * 100,
        })
    return pd.DataFrame(rows)


gc_per_class = per_class_breakdown(gradcam_results, EVAL_WNIDS, EVAL_CLASS_INDICES)
cam_per_class = per_class_breakdown(cam_results, EVAL_WNIDS, EVAL_CLASS_INDICES)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, df, title in zip(axes,
                          [gc_per_class, cam_per_class],
                          ['Grad-CAM', 'Modified CAM']):
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w/2, df['Cls Acc (%)'], w, label='Classification', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, df['Loc Top-1 Acc (%)'], w, label='Localisation',   color='coral',     alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df['WNID'], rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{title} — Per-Class Performance')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('per_class_breakdown.png', dpi=150, bbox_inches='tight')
plt.show()



--------------------------------------------------------------
# Summary Bar Chart — Replicating Table 1
----------------------------------------------------------------

methods   = ['Grad-CAM\n(Ours)', 'CAM\n(Modified, Ours)',
             'Grad-CAM\n[Paper]', 'CAM\n[Paper]']
top1_errs = [
    gradcam_results['loc_top1_err'],
    cam_results['loc_top1_err'],
    56.51,
    57.20,
]
top5_errs = [
    gradcam_results['loc_top5_err'],
    cam_results['loc_top5_err'],
    46.41,
    45.14,
]

x = np.arange(len(methods))
w = 0.35
colors_top1 = ['#2196F3', '#FF9800', '#2196F3', '#FF9800']
colors_top5 = ['#1565C0', '#E65100', '#1565C0', '#E65100']

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - w/2, top1_errs, w, label='Top-1 Loc Error', color=colors_top1, alpha=0.85)
bars2 = ax.bar(x + w/2, top5_errs, w, label='Top-5 Loc Error', color=colors_top5, alpha=0.85)

# Annotate bar values
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

ax.axvline(x=1.5, color='grey', linestyle='--', linewidth=1.5, alpha=0.6)
ax.text(0.8, ax.get_ylim()[1]*0.97, 'Our results', ha='center', fontsize=9, color='grey')
ax.text(2.2, ax.get_ylim()[1]*0.97, 'Paper results', ha='center', fontsize=9, color='grey')

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel('Localisation Error (%) — lower is better')
ax.set_title('Grad-CAM vs CAM: Top-1 and Top-5 Localisation Error\n'
             '(Replicating Table 1 of Selvaraju et al., 2017)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim(0, max(top1_errs + top5_errs) * 1.15)

plt.tight_layout()
plt.savefig('localization_error_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nFinal Summary:')
print(f'  Grad-CAM  Top-1 Loc Error: {gradcam_results["loc_top1_err"]:.2f}%  (Paper: 56.51%)')
print(f'  CAM       Top-1 Loc Error: {cam_results["loc_top1_err"]:.2f}%  (Paper: 57.20%)')
print(f'  --> Grad-CAM better: {cam_results["loc_top1_err"] > gradcam_results["loc_top1_err"]}')
