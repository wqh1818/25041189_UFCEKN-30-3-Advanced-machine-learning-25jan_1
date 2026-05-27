-------------------------------------------------
# Install Dependencies, import, and global configurations
-------------------------------------------------

!pip install timm -q

import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from PIL import Image
from copy import deepcopy

from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import timm
import cv2

warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 3
EPOCHS = 15
LR = 1e-4
PATIENCE = 4
HEATMAP_THRESHOLD = 25
ROLLOUT_DISCARD = 0.9

CLASS_NAMES = ['COVID-19', 'Lung Opacity', 'Normal']
CLS_TO_IDX  = {'covid19': 0, 'lung_opacity': 1, 'normal': 2}

print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"timm   : {timm.__version__}")




-------------------------------------------------
# 1. Mount Drive & Extract Dataset
-------------------------------------------------

from google.colab import drive
drive.mount('/content/drive')

zip_path    = "/content/drive/MyDrive/AML project/datasets/datasets_normalandmasks.zip"
extract_dir = "/content/local_dataset/"
base_path   = "/content/local_dataset/datasets"

if os.path.exists(base_path) and len(os.listdir(base_path)) > 0:
    print("Dataset already extracted — skipping unzip.")
else:
    os.makedirs(extract_dir, exist_ok=True)
    print("Extracting dataset …")
    os.system(f'unzip -q "{zip_path}" -d "{extract_dir}"')
    print("Extraction complete!")

print(f"\nData ready at : {base_path}")
print("Folders found :", sorted(os.listdir(base_path)))



-------------------------------------------------
# 2. Detect Folder Structure
-------------------------------------------------

all_folders = sorted(os.listdir(base_path))
image_folders = {'covid19': None, 'lung_opacity': None, 'normal': None}
mask_folders = {'covid19': None, 'lung_opacity': None, 'normal': None}

for folder in all_folders:
    fl = folder.lower()
    target = mask_folders if 'mask' in fl else image_folders
    if 'covid' in fl:
        target['covid19'] = folder
    elif 'opacity' in fl:
        target['lung_opacity'] = folder
    elif 'normal' in fl:
        target['normal'] = folder

print("Image folders:", image_folders)
print("Mask folders:", mask_folders)

for cls in ['covid19', 'lung_opacity', 'normal']:
    img_dir = os.path.join(base_path, image_folders[cls])
    msk_dir = os.path.join(base_path, mask_folders[cls])
    n_img = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    n_msk = len([f for f in os.listdir(msk_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"  {cls}: {n_img} images, {n_msk} masks")



-------------------------------------------------
# 3. Dataset Class, Transforms and DataLoaders
-------------------------------------------------

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

def build_samples(base_path, image_folders, mask_folders):
    samples = []
    missing = 0
    for cls, idx in CLS_TO_IDX.items():
        img_dir = os.path.join(base_path, image_folders[cls])
        msk_dir = os.path.join(base_path, mask_folders[cls])
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(img_dir, fname)
            mask_path = os.path.join(msk_dir, fname)
            if os.path.exists(mask_path):
                samples.append((img_path, mask_path, idx))
            else:
                missing += 1
    if missing:
        print(f"WARNING: {missing} images had no matching mask and were skipped.")
    random.shuffle(samples)
    return samples


class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        img_lab[:, :, 0] = self.clahe.apply(img_lab[:, :, 0])
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(img_rgb)


clahe = CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8))


class JointTransform:
    #Applies transforms to image and mask jointly so they always stay aligned.

    def __init__(self, augment=False):
        self.augment = augment
        self.clahe = CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8))
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, image, mask):
        # resize both to the same size
        image = TF.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = TF.resize(
            mask, (IMG_SIZE, IMG_SIZE),
            interpolation=transforms.InterpolationMode.NEAREST
        )

        # CLAHE on image only
        image = self.clahe(image)

        # spatial augmentations applied jointly to both
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random rotation, same angle for both
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            # Random affine, same params for both
            affine_params = transforms.RandomAffine.get_params(
                degrees=(0, 0),
                translate=(0.05, 0.05),
                scale_ranges=(0.95, 1.05),
                shears=None,
                img_size=(IMG_SIZE, IMG_SIZE)
            )
            image = TF.affine(image, *affine_params)
            mask = TF.affine(
                mask, *affine_params,
                interpolation=transforms.InterpolationMode.NEAREST
            )

            # ColorJitter on image only
            image = self.color_jitter(image)

        # convert to tensors
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # normalise image only
        image = self.normalize(image)

        # binarise mask
        mask = (mask > 0.5).float()

        return image, mask


class LungDataset(Dataset):
    # Updated to use JointTransform so image and mask always receive the same spatial operations.
    def __init__(self, samples, joint_transform=None):
        self.samples = samples
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        return image, mask, label, img_path


# Train uses augmentation, val and test do not
train_transform = JointTransform(augment=True)
eval_transform = JointTransform(augment=False)

# Build and split samples 80/10/10
all_samples = build_samples(base_path, image_folders, mask_folders)
n_total = len(all_samples)
n_train = int(0.80 * n_total)
n_val = int(0.10 * n_total)
n_test = n_total - n_train - n_val

train_samples = all_samples[:n_train]
val_samples = all_samples[n_train : n_train + n_val]
test_samples = all_samples[n_train + n_val:]

print(f"Total: {n_total} | Train: {n_train} | Val: {n_val} | Test: {n_test}")

train_ds = LungDataset(train_samples, joint_transform=train_transform)
val_ds = LungDataset(val_samples, joint_transform=eval_transform)
test_ds = LungDataset(test_samples, joint_transform=eval_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print("DataLoaders ready.")



def show_alignment_check(train_loader, n=4):

    # Pulls a batch from the train loader (which uses augmentation) and shows the image, mask and overlay side by side.
    def denorm(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    imgs, masks, labels, _ = next(iter(train_loader))

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))

    for i in range(n):
        img_np = denorm(imgs[i])
        mask_np = masks[i].squeeze().numpy()

        overlay = img_np.copy()
        overlay[:, :, 1] = np.where(mask_np > 0.5, 1.0, overlay[:, :, 1])

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(CLASS_NAMES[labels[i].item()], fontsize=9)
        axes[0, i].axis('off')

        axes[1, i].imshow(mask_np, cmap='gray')
        axes[1, i].set_title('Mask', fontsize=9)
        axes[1, i].axis('off')

        axes[2, i].imshow(overlay)
        axes[2, i].set_title('Overlay (check alignment)', fontsize=9)
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel('Image', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Mask', fontsize=10, fontweight='bold')
    axes[2, 0].set_ylabel('Overlay', fontsize=10, fontweight='bold')

    fig.suptitle('Alignment Check - Image and Mask Should Match in Overlay',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('alignment_check.png', dpi=150, bbox_inches='tight')
    plt.show()


def show_clahe_comparison(n=4):

    #Shows original vs CLAHE-enhanced images side by side to confirm contrast enhancement is working on dim images.
    sample_paths = [test_samples[i][0] for i in range(n)]
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for i, path in enumerate(sample_paths):
        img_pil = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
        clahe_pil = clahe(img_pil)

        axes[0, i].imshow(np.array(img_pil))
        axes[0, i].set_title('Original', fontsize=10)
        axes[0, i].axis('off')

        axes[1, i].imshow(np.array(clahe_pil))
        axes[1, i].set_title('After CLAHE', fontsize=10)
        axes[1, i].axis('off')

    fig.suptitle('CLAHE Contrast Enhancement - Before vs After',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('clahe_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


show_clahe_comparison(n=4)
show_alignment_check(train_loader, n=4)




-------------------------------------------------
# 4. Check Visualisation
-------------------------------------------------
def denorm(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

imgs, masks, labels, _ = next(iter(train_loader))

fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for i in range(6):
    img_np = denorm(imgs[i])
    mask_np = masks[i].squeeze().numpy()

    axes[0, i].imshow(img_np)
    axes[0, i].set_title(CLASS_NAMES[labels[i].item()], fontsize=9)
    axes[0, i].axis('off')

    axes[1, i].imshow(mask_np, cmap='gray')
    axes[1, i].set_title('Lung Mask', fontsize=9)
    axes[1, i].axis('off')

    overlay = img_np.copy()
    overlay[:, :, 1] = np.where(mask_np > 0.5, 1.0, overlay[:, :, 1])
    axes[2, i].imshow(overlay)
    axes[2, i].set_title('Overlay', fontsize=9)
    axes[2, i].axis('off')

fig.suptitle('Dataset Samples | Lung Masks | Overlay', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_visualisation.png', dpi=150, bbox_inches='tight')
plt.show()



-------------------------------------------------
# 5. Training and Evaluation Utilities
-------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, masks, labels, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, masks, labels, _ in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, patience=PATIENCE, model_name='model'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_weights = None
    no_improve = 0

    print(f"\nTraining {model_name}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(vl_acc)

        print(f"Epoch {epoch}/{epochs} | Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f}, Acc: {vl_acc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_weights = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_weights)
    print(f"Best val loss: {best_val_loss:.4f}")
    return model, history


def plot_training_curves(history, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history['train_loss']) + 1)

    ax1.plot(ep, history['train_loss'], 'b-o', markersize=4, label='Train')
    ax1.plot(ep, history['val_loss'], 'r-o', markersize=4, label='Val')
    ax1.set_title(f'{model_name} - Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ep, history['train_acc'], 'b-o', markersize=4, label='Train')
    ax2.plot(ep, history['val_acc'], 'r-o', markersize=4, label='Val')
    ax2.set_title(f'{model_name} - Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{model_name}_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()



-------------------------------------------------
# 6. Build and Train VGG16
-------------------------------------------------

def build_vgg16(num_classes=NUM_CLASSES):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Freeze early layers up to conv4_1, fine-tune deeper layers
    for i, layer in enumerate(model.features):
        if i < 17:
            for p in layer.parameters():
                p.requires_grad = False

    # Replace the output head for 3 classes
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model.to(DEVICE)


vgg16_model = build_vgg16()
trainable = sum(p.numel() for p in vgg16_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in vgg16_model.parameters())
print(f"VGG16 - Trainable params: {trainable:,} / {total:,}")

vgg16_model, vgg16_history = train_model(
    vgg16_model, train_loader, val_loader,
    epochs=EPOCHS, lr=LR, patience=PATIENCE, model_name='VGG16'
)
torch.save(vgg16_model.state_dict(), 'vgg16_best.pth')
plot_training_curves(vgg16_history, 'VGG16')



-------------------------------------------------
# 7. Build and Train ViT
-------------------------------------------------

def build_vit(num_classes=NUM_CLASSES):
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze the last 4 transformer blocks, the head and final norm
    for name, p in model.named_parameters():
        if any(f'blocks.{i}' in name for i in [8, 9, 10, 11]):
            p.requires_grad = True
        if 'head' in name or name.startswith('norm'):
            p.requires_grad = True

    return model.to(DEVICE)


vit_model = build_vit()
trainable = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in vit_model.parameters())
print(f"ViT-B/16 - Trainable params: {trainable:,} / {total:,}")

# ViT benefits from a slightly lower LR than VGG16 during fine-tuning
vit_model, vit_history = train_model(
    vit_model, train_loader, val_loader,
    epochs=EPOCHS, lr=LR * 0.5, patience=PATIENCE, model_name='ViT-B16'
)
torch.save(vit_model.state_dict(), 'vit_best.pth')
plot_training_curves(vit_history, 'ViT-B16')

-------------------------------------------------
# 8. Standard Grad-CAM for VGG16
-------------------------------------------------

def disable_inplace_relu(model):
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

# Disable inplace ReLU across the entire VGG16 model
disable_inplace_relu(vgg16_model)
print("Inplace ReLU disabled on VGG16.")


class GradCAM_VGG:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None

        target = model.features[28]

        target.register_forward_hook(
            lambda m, inp, out: setattr(self, 'activations', out.detach().clone())
        )
        target.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach().clone())
        )

    def generate(self, img_tensor, class_idx=None):
        self.model.eval()
        img_tensor = img_tensor.to(DEVICE)

        output = self.model(img_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam, class_idx


# Re-instantiate with the fixed class
gradcam_vgg = GradCAM_VGG(vgg16_model)
print("GradCAM_VGG ready.")



-------------------------------------------------
# Attention Rollout for ViT & Raw Gradient Grad-CAM for ViT
-------------------------------------------------

class AttentionRollout:
    def __init__(self, model, discard_ratio=ROLLOUT_DISCARD):
        self.model = model
        self.discard_ratio = discard_ratio

    def _disable_fused_attn(self):
        for block in self.model.blocks:
            block.attn.fused_attn = False

    def _enable_fused_attn(self):
        for block in self.model.blocks:
            block.attn.fused_attn = True

    def _get_attention_weights(self, img_tensor):
        attn_weights = []
        hooks = []

        def make_hook():
            def hook_fn(module, input, output):
                attn_weights.append(input[0].detach().clone())
            return hook_fn

        for block in self.model.blocks:
            hooks.append(block.attn.attn_drop.register_forward_hook(make_hook()))

        # Temporarily clear any foreign hooks on blocks[-1] so they don't fire inside torch.no_grad() and crash on retain_grad()
        saved_hooks = dict(self.model.blocks[-1]._forward_hooks)
        self.model.blocks[-1]._forward_hooks.clear()

        with torch.no_grad():
            output = self.model(img_tensor)

        # Restore the saved hooks on blocks[-1]
        self.model.blocks[-1]._forward_hooks.update(saved_hooks)

        for h in hooks:
            h.remove()

        return attn_weights, output

    def generate(self, img_tensor, class_idx=None):
        self.model.eval()
        img_tensor = img_tensor.to(DEVICE)

        self._disable_fused_attn()
        attn_weights, output = self._get_attention_weights(img_tensor)
        self._enable_fused_attn()

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        if len(attn_weights) == 0:
            raise RuntimeError("No attention weights captured.")

        n_tokens = attn_weights[0].size(-1)
        rollout = torch.eye(n_tokens, device=DEVICE).unsqueeze(0)

        for attn in attn_weights:
            attn_avg = attn.mean(dim=1)

            flat = attn_avg.view(1, -1)
            threshold = torch.quantile(flat, self.discard_ratio, dim=-1)
            attn_avg = torch.where(
                attn_avg < threshold.unsqueeze(-1).unsqueeze(-1),
                torch.zeros_like(attn_avg),
                attn_avg
            )

            attn_avg = 0.5 * attn_avg + 0.5 * torch.eye(n_tokens, device=DEVICE)
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            rollout = attn_avg @ rollout

        mask = rollout[0, 0, 1:].cpu().numpy()
        n_patches = int(round(mask.shape[0] ** 0.5))
        mask = mask.reshape(n_patches, n_patches)

        if mask.max() > 0:
            mask = mask / mask.max()
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        return mask, class_idx


class GradCAM_ViT:
    def __init__(self, model):
        self.model = model

    def generate(self, img_tensor, class_idx=None):
        self.model.eval()

        img_input = img_tensor.to(DEVICE).clone().requires_grad_(True)

        model_output = self.model(img_input)
        if class_idx is None:
            class_idx = model_output.argmax(dim=1).item()

        self.model.zero_grad()
        model_output[0, class_idx].backward()

        saliency = img_input.grad.abs().mean(dim=1, keepdim=True)

        # Pool into 14x14 patch grid to match ViT-B/16 patch resolution
        # Each 16x16 pixel region maps to one patch token
        import torch.nn.functional as F
        cam = F.avg_pool2d(saliency, kernel_size=16, stride=16)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam, class_idx


# Clear any leftover hooks on blocks[-1]
vit_model.blocks[-1]._forward_hooks.clear()

gradcam_vit = GradCAM_ViT(vit_model)
print("GradCAM_ViT (input gradient) ready.")

# Sanity check
test_img, test_mask, test_label, _ = test_ds[0]
hmap, pred = gradcam_vit.generate(test_img.unsqueeze(0))
mask_np = test_mask.squeeze().numpy()
ratio = (hmap * mask_np).sum() / (hmap.sum() + 1e-8)
print(f"Heatmap max : {hmap.max():.4f}")
print(f"Attn ratio  : {ratio:.4f}")
print(f"Predicted   : {CLASS_NAMES[pred]}")
print("Fix confirmed." if hmap.max() > 0 else "Still broken - heatmap is zero.")



-------------------------------------------------
# 11. IoU and Attention Ratio Utilities
-------------------------------------------------


def smooth_heatmap(heatmap, sigma=20):
    smoothed = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if smoothed.max() > 0:
        smoothed = smoothed / smoothed.max()
    return smoothed


def binarise_heatmap(heatmap, percentile=HEATMAP_THRESHOLD):
    heatmap = smooth_heatmap(heatmap)
    threshold = np.percentile(heatmap, percentile)
    return (heatmap >= threshold).astype(np.float32)


def compute_iou(binary_heatmap, binary_mask):
    intersection = (binary_heatmap * binary_mask).sum()
    union = np.clip(binary_heatmap + binary_mask, 0, 1).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_attention_ratio(heatmap, binary_mask):
    heatmap = smooth_heatmap(heatmap)
    inside = (heatmap * binary_mask).sum()
    total = heatmap.sum()
    if total == 0:
        return 0.0
    return float(inside / total)




-------------------------------------------------
# 12. Full Test-Set Evaluation
-------------------------------------------------

def run_evaluation(model, explainer, test_loader, method_name):
    model.eval()
    all_preds = []
    all_labels = []
    iou_per_class = {i: [] for i in range(NUM_CLASSES)}
    ratio_per_class = {i: [] for i in range(NUM_CLASSES)}

    print(f"\n[{method_name}] Running evaluation...")

    for batch_idx, (imgs, masks, labels, _) in enumerate(test_loader):
        labels_np = labels.numpy()
        masks_np = masks.squeeze(1).numpy()

        with torch.no_grad():
            outputs = model(imgs.to(DEVICE))
            preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels_np.tolist())

        for i in range(imgs.size(0)):
            heatmap, _ = explainer.generate(imgs[i].unsqueeze(0))
            binary_hmap = binarise_heatmap(heatmap)
            binary_mask = (masks_np[i] > 0.5).astype(np.float32)

            cls = int(labels_np[i])
            iou_per_class[cls].append(compute_iou(binary_hmap, binary_mask))
            ratio_per_class[cls].append(compute_attention_ratio(heatmap, binary_mask))

        if (batch_idx + 1) % 5 == 0:
            done = min((batch_idx + 1) * BATCH_SIZE, len(test_loader.dataset))
            print(f"  {done} / {len(test_loader.dataset)} samples processed")

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_cls = f1_score(all_labels, all_preds, average=None)

    mean_iou_cls = {i: np.mean(v) for i, v in iou_per_class.items()}
    mean_ratio_cls = {i: np.mean(v) for i, v in ratio_per_class.items()}
    iou_overall = np.mean([v for vals in iou_per_class.values() for v in vals])
    ratio_overall = np.mean([v for vals in ratio_per_class.values() for v in vals])

    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  Mean IoU: {iou_overall:.4f}")
    print(f"  Attn Ratio: {ratio_overall:.4f}")

    return {
        'method_name': method_name,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_per_class': f1_cls,
        'iou_per_class': mean_iou_cls,
        'ratio_per_class': mean_ratio_cls,
        'iou_overall': iou_overall,
        'ratio_overall': ratio_overall,
        'all_preds': all_preds,
        'all_labels': all_labels,
    }


# Instantiate all three explainers
gradcam_vgg = GradCAM_VGG(vgg16_model)
attn_rollout = AttentionRollout(vit_model, discard_ratio=ROLLOUT_DISCARD)
gradcam_vit = GradCAM_ViT(vit_model)

# Run evaluations
print("\nEVALUATION - All Methods")
print("-" * 50)

vgg_results = run_evaluation(vgg16_model, gradcam_vgg, test_loader, 'VGG16 + Grad-CAM')
rollout_results = run_evaluation(vit_model, attn_rollout, test_loader, 'ViT + Attn Rollout (Primary)')
vitcam_results = run_evaluation(vit_model, gradcam_vit, test_loader, 'ViT + Grad-CAM (Secondary)')



-------------------------------------------------
# 13. Results Summary Table
-------------------------------------------------

rows = []
for res in [vgg_results, rollout_results, vitcam_results]:
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        rows.append({
            'Method': res['method_name'],
            'Class': cls_name,
            'Accuracy': round(res['accuracy'], 4),
            'F1 (macro)': round(res['f1_macro'], 4),
            'F1 (class)': round(float(res['f1_per_class'][cls_idx]), 4),
            'Mean IoU': round(res['iou_per_class'][cls_idx], 4),
            'Attn Ratio': round(res['ratio_per_class'][cls_idx], 4),
        })

df = pd.DataFrame(rows)
print("\nRESULTS SUMMARY")
print("-" * 50)
print(df.to_string(index=False))

print("\nOverall (across all classes):")
for res in [vgg_results, rollout_results, vitcam_results]:
    print(f"  {res['method_name']}")
    print(f"    Acc: {res['accuracy']:.4f} | F1: {res['f1_macro']:.4f} | IoU: {res['iou_overall']:.4f} | Ratio: {res['ratio_overall']:.4f}")




-------------------------------------------------
# 14, Confusion Matrices
-------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

titles = [
    'VGG16 + Grad-CAM',
    'ViT + Attention Rollout (Primary)',
    'ViT + Grad-CAM (Secondary)'
]

for ax, res, title in zip(axes, [vgg_results, rollout_results, vitcam_results], titles):
    cm = confusion_matrix(res['all_labels'], res['all_preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={'size': 12})
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)

plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()



-------------------------------------------------
# 15. Per-Class Metric Bar Charts
-------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

results_list = [vgg_results, rollout_results, vitcam_results]
bar_labels = ['VGG16\nGrad-CAM', 'ViT\nRollout', 'ViT\nGrad-CAM']
colours = ['#4e79a7', '#f28e2b', '#e15759']

metrics_cfg = [
    ('f1_per_class', 'F1-Score per Class'),
    ('iou_per_class', 'Mean IoU per Class'),
    ('ratio_per_class', 'Attention Ratio per Class'),
]

x = np.arange(len(CLASS_NAMES))
width = 0.22

for ax, (key, title) in zip(axes, metrics_cfg):
    for i, (res, bar_label, colour) in enumerate(zip(results_list, bar_labels, colours)):
        if key == 'f1_per_class':
            vals = [float(res['f1_per_class'][cls_idx]) for cls_idx in range(NUM_CLASSES)]
        else:
            vals = [res[key][cls_idx] for cls_idx in range(NUM_CLASSES)]

        bars = ax.bar(x + (i - 1) * width, vals, width, label=bar_label, color=colour, alpha=0.87)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=7
            )

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Per-Class Metrics - VGG16 vs ViT (Rollout) vs ViT (Grad-CAM)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('per_class_metrics.png', dpi=150, bbox_inches='tight')
plt.show()



-------------------------------------------------
# 16. Heatmap Gallery
-------------------------------------------------

def make_overlay(img_tensor, heatmap):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_np = ((img_tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    hmap_u8 = (heatmap * 255).astype(np.uint8)
    hmap_col = cv2.cvtColor(cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.55, hmap_col, 0.45, 0)
    return img_np, overlay


def visualise_gallery(test_loader, n_per_class=2):
    class_samples = {i: [] for i in range(NUM_CLASSES)}
    for imgs, masks, labels, _ in test_loader:
        for i in range(imgs.size(0)):
            cls = labels[i].item()
            if len(class_samples[cls]) < n_per_class:
                class_samples[cls].append((imgs[i], masks[i].squeeze()))
        if all(len(v) >= n_per_class for v in class_samples.values()):
            break

    n_rows = NUM_CLASSES * n_per_class
    fig, axes = plt.subplots(n_rows, 5, figsize=(22, n_rows * 4.2))

    col_headers = [
        'Original X-Ray',
        'Lung Mask (GT)',
        'VGG16 Grad-CAM',
        'ViT Attention Rollout',
        'ViT Grad-CAM (secondary)'
    ]

    row = 0
    for cls_idx in range(NUM_CLASSES):
        for si in range(n_per_class):
            img_t, mask_t = class_samples[cls_idx][si]
            img_single = img_t.unsqueeze(0)

            hmap_vgg, _ = gradcam_vgg.generate(img_single)
            hmap_rollout, _ = attn_rollout.generate(img_single)
            hmap_vitcam, _ = gradcam_vit.generate(img_single)

            img_np, ov_vgg = make_overlay(img_t, hmap_vgg)
            _, ov_rollout = make_overlay(img_t, hmap_rollout)
            _, ov_vitcam = make_overlay(img_t, hmap_vitcam)

            axes[row, 0].imshow(img_np)
            axes[row, 0].set_ylabel(f'{CLASS_NAMES[cls_idx]}\nsample {si + 1}', fontsize=9, fontweight='bold')
            axes[row, 1].imshow(mask_t.numpy(), cmap='gray')
            axes[row, 2].imshow(ov_vgg)
            axes[row, 3].imshow(ov_rollout)
            axes[row, 4].imshow(ov_vitcam)

            for ax in axes[row]:
                ax.axis('off')
            row += 1

    for col_idx, hdr in enumerate(col_headers):
        axes[0, col_idx].set_title(hdr, fontsize=10, fontweight='bold', pad=6)

    plt.suptitle('Heatmap Gallery - All Three Explanation Methods', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('gradcam_gallery.png', dpi=150, bbox_inches='tight')
    plt.show()


visualise_gallery(test_loader, n_per_class=2)



-------------------------------------------------
# 17. Attention Alignment vs Classification Performance Scatter
-------------------------------------------------


fig, ax = plt.subplots(figsize=(9, 7))
colours_cls = ['#e63946', '#457b9d', '#2a9d8f']
markers = {
    'VGG16 + Grad-CAM': 'o',
    'ViT + Attn Rollout (Primary)': 'D',
    'ViT + Grad-CAM (Secondary)': 'X'
}

for res in [vgg_results, rollout_results, vitcam_results]:
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        iou = res['iou_per_class'][cls_idx]
        f1 = float(res['f1_per_class'][cls_idx])
        marker = markers.get(res['method_name'], 'o')
        ax.scatter(iou, f1, color=colours_cls[cls_idx], marker=marker,
                   s=120, zorder=5, edgecolors='white', linewidths=0.8)
        ax.annotate(
            f"{res['method_name'].split(' ')[0]}\n{cls_name}",
            (iou, f1), textcoords='offset points', xytext=(6, 3),
            fontsize=7, color='#333333'
        )

ax.set_xlabel('Mean IoU (heatmap vs lung mask)', fontsize=12)
ax.set_ylabel('F1-Score per class', fontsize=12)
ax.set_title('Attention Alignment vs Classification Performance\nper Class and Explanation Method', fontsize=12, fontweight='bold')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)

legend_method = [
    mlines.Line2D([], [], marker='o', color='gray', linestyle='None', markersize=10, label='VGG16 Grad-CAM'),
    mlines.Line2D([], [], marker='D', color='gray', linestyle='None', markersize=10, label='ViT Attention Rollout'),
    mlines.Line2D([], [], marker='X', color='gray', linestyle='None', markersize=10, label='ViT Grad-CAM'),
]
legend_cls = [
    mlines.Line2D([], [], marker='s', color=colours_cls[i], linestyle='None', markersize=10, label=CLASS_NAMES[i])
    for i in range(NUM_CLASSES)
]
ax.legend(handles=legend_method + legend_cls, loc='lower right', fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig('iou_vs_f1_scatter.png', dpi=150, bbox_inches='tight')
plt.show()



-------------------------------------------------
# 18. Rollout vs Grad-CAM IoU Agreement on ViT
-------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(CLASS_NAMES))
width = 0.25

vgg_iou = [vgg_results['iou_per_class'][i] for i in range(NUM_CLASSES)]
rollout_iou = [rollout_results['iou_per_class'][i] for i in range(NUM_CLASSES)]
vitcam_iou = [vitcam_results['iou_per_class'][i] for i in range(NUM_CLASSES)]

b1 = ax.bar(x - width, vgg_iou, width, label='VGG16 Grad-CAM', color='#4e79a7', alpha=0.87)
b2 = ax.bar(x, rollout_iou, width, label='ViT Attn Rollout', color='#f28e2b', alpha=0.87)
b3 = ax.bar(x + width, vitcam_iou, width, label='ViT Grad-CAM (secondary)', color='#e15759', alpha=0.87)

for bars in [b1, b2, b3]:
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{bar.get_height():.3f}',
            ha='center', va='bottom', fontsize=9
        )

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=11)
ax.set_ylabel('Mean IoU', fontsize=12)
ax.set_ylim(0, 1.0)
ax.set_title('IoU Comparison - Do Rollout and Grad-CAM Agree?\n(ViT vs VGG16 Baseline)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('rollout_vs_gradcam_iou.png', dpi=150, bbox_inches='tight')
plt.show()



-------------------------------------------------
# 19. Final Summary
-------------------------------------------------

print("\nFINAL RESULTS SUMMARY - Part C")
print("-" * 50)

all_results = [vgg_results, rollout_results, vitcam_results]

for res in all_results:
    print(f"\n{res['method_name']}")
    print(f"  Accuracy : {res['accuracy']:.4f}")
    print(f"  F1 macro : {res['f1_macro']:.4f}")
    print(f"  Mean IoU : {res['iou_overall']:.4f}")
    print(f"  Attn Ratio: {res['ratio_overall']:.4f}")
    print(f"  Per class:")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        f1 = float(res['f1_per_class'][cls_idx])
        iou = res['iou_per_class'][cls_idx]
        ratio = res['ratio_per_class'][cls_idx]
        print(f"    {cls_name}: F1={f1:.4f}, IoU={iou:.4f}, Ratio={ratio:.4f}")

print("\n")
print("Saved figures:")
print("  sample_visualisation.png    - dataset sanity check")
print("  VGG16_training_curves.png   - VGG16 loss and accuracy")
print("  ViT-B16_training_curves.png - ViT loss and accuracy")
print("  confusion_matrices.png      - all three confusion matrices")
print("  per_class_metrics.png       - F1, IoU, Attention Ratio bar charts")
print("  gradcam_gallery.png         - side-by-side heatmap gallery")
print("  iou_vs_f1_scatter.png       - alignment vs performance scatter")
print("  rollout_vs_gradcam_iou.png  - Rollout vs Grad-CAM IoU comparison")
