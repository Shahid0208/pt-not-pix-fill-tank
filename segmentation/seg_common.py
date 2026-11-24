"""Dataset, model and metrics shared by train_seg.py and evaluate_seg.py.

Segmentation uses the CNN only: MobileNetV3-Small U-Net (`baselines/mobilenet_unet.MobileNetUNetSkip`),
ImageNet-pretrained, on ImageNet-normalised 128 px crops. resnet18_unet is the matched model (the keypoint model's
backbone; use it with 256 px crops over 2 rim diameters)."""
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from baselines.mobilenet_unet import MobileNetUNetSkip      # noqa: E402
from baselines.resnet_unet import ResNet18UNet              # noqa: E402

NUM_CLASSES, IGNORE = 3, 255
CLASS_NAMES = ["background", "ground_shadow", "inner_shadow"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]
MODEL = "mobilenet_unet"
MODELS = {"mobilenet_unet": MobileNetUNetSkip,
          "resnet18_unet": ResNet18UNet}   


def build_model(name=MODEL):
    return MODELS[name](num_classes=NUM_CLASSES, pretrained=True)


def encode_input(img_rgb):
    """HWC RGB float image in [0, 1] -> ImageNet-normalised [C, H, W] tensor."""
    chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray((chw - IMAGENET_MEAN) / IMAGENET_STD, dtype=np.float32))


class SegDataset(Dataset):
    """Crops and masks from prepare_synthetic_seg.py."""

    def __init__(self, root, split, augment, names=None, seed=0):
        self.root, self.augment = root, augment
        self.names = names if names is not None else \
            [l.strip() for l in open(os.path.join(root, f"{split}.txt")) if l.strip()]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.names)

    def meta(self, i):
        return json.load(open(os.path.join(self.root, "meta", self.names[i] + ".json")))

    def _augment(self, img, msk):
        r = self.rng
        if r.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        if r.random() < 0.5:
            img, msk = img[::-1], msk[::-1]
        k = int(r.integers(4))
        img, msk = np.rot90(img, k), np.rot90(msk, k)
        img = img * r.uniform(0.8, 1.2) + r.uniform(-0.05, 0.05)                       # gain, offset
        m = img.mean()
        img = (img - m) * r.uniform(0.8, 1.2) + m                                      # contrast
        if r.random() < 0.3:
            img = cv2.GaussianBlur(np.ascontiguousarray(img), (0, 0), r.uniform(0.4, 1.0))
        img = img + r.normal(0, r.uniform(0, 0.02), img.shape)                         # sensor noise
        return np.clip(img, 0, 1).astype(np.float32), np.ascontiguousarray(msk)

    def __getitem__(self, i):
        n = self.names[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.root, "images", n + ".png")), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        msk = cv2.imread(os.path.join(self.root, "masks", n + ".png"), cv2.IMREAD_GRAYSCALE)
        if self.augment:
            img, msk = self._augment(img, msk)
        x = encode_input(img)
        return x, torch.from_numpy(msk.astype(np.int64)), i


class SegLoss(nn.Module):
    """Cross-entropy (class-weighted) + soft Dice over the two shadow classes; IGNORE pixels excluded."""

    def __init__(self, class_weights=(1.0, 1.0, 3.0), dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights), ignore_index=IGNORE)
        self.dice_weight = dice_weight

    def forward(self, logits, y):
        self.ce.weight = self.ce.weight.to(logits.device)
        ce = self.ce(logits, y)
        valid = (y != IGNORE)
        p = logits.softmax(1)
        dice = 0.0
        for c in (1, 2):
            t = ((y == c) & valid).float()
            pc = p[:, c] * valid
            inter = (pc * t).sum()
            dice = dice + 1 - (2 * inter + 1) / (pc.sum() + t.sum() + 1)
        return ce + self.dice_weight * dice / 2


def confusion(pred, y):
    """3x3 confusion counts (rows truth, cols prediction), IGNORE excluded."""
    v = y != IGNORE
    return np.bincount(y[v] * NUM_CLASSES + pred[v], minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def iou_from_confusion(cm):
    inter = np.diag(cm).astype(float)
    union = cm.sum(0) + cm.sum(1) - inter
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def boundary_f1(pred_c, gt_c, valid, tol=1):
    """Boundary F1 of one binary class with a `tol`-pixel tolerance. nan if both are empty."""
    def edge(m):
        m = m.astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool) & valid
    pe, ge = edge(pred_c), edge(gt_c)
    if not pe.any() and not ge.any():
        return np.nan
    if not pe.any() or not ge.any():
        return 0.0
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    prec = (pe & cv2.dilate(ge.astype(np.uint8), k).astype(bool)).sum() / pe.sum()
    rec = (ge & cv2.dilate(pe.astype(np.uint8), k).astype(bool)).sum() / ge.sum()
    return 0.0 if prec + rec == 0 else float(2 * prec * rec / (prec + rec))
