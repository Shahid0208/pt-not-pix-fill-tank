"""PyTorch dataset: sun-aligned context crop -> 256x256 input, 65 Gaussian heatmaps."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import geometry_kp as G

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_records(root, split):
    root = Path(root)
    names = set((root / f"{split}.txt").read_text().split())
    recs = [r for r in json.load(open(root / "keypoints.json")) if r["image"] in names]
    assert len(recs) == len(names), f"{split}: {len(names)} listed, {len(recs)} found"
    return recs


class KeypointDataset(Dataset):
    def __init__(self, root, split, cfg, train, rot_offset_deg=0.0):
        self.root = Path(root)
        self.recs = load_records(root, split)
        self.train = train
        self.cfg = cfg
        self.out = cfg["input_size"]
        self.hm = cfg["heatmap_size"]
        self.stride = self.out / self.hm
        self.sigma = cfg["sigma"]
        self.flip_idx = G.flip_index()
        self.rot_offset = rot_offset_deg  # negative control: wrong sun direction
        ys, xs = np.mgrid[0:self.hm, 0:self.hm].astype(np.float32)
        self.grid = (xs, ys)

    def __len__(self):
        return len(self.recs)

    def _ctx_to_out(self, rng):
        """Affine context crop (320 = 5R) -> model input (256 = 4R at scale 1)."""
        cfg = self.cfg
        ctx = cfg["ctx_size"]
        scale, rot, tx, ty, flip = 1.0, self.rot_offset, 0.0, 0.0, False
        if self.train:
            scale = 1.0 + rng.uniform(-cfg["aug_scale"], cfg["aug_scale"])
            rot += rng.uniform(-cfg["aug_rot_deg"], cfg["aug_rot_deg"])
            r_px = ctx / (2 * cfg["ctx_factor"])  # one tank radius in context pixels
            tx, ty = rng.uniform(-cfg["aug_shift"], cfg["aug_shift"], 2) * r_px
            flip = rng.random() < cfg["aug_flip"]
        a = math.radians(rot)
        L = scale * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        if flip:
            L = np.diag([-1.0, 1.0]) @ L
        t = np.array([self.out / 2 + tx, self.out / 2 + ty]) - L @ np.array([ctx / 2, ctx / 2])
        return np.hstack([L, t[:, None]]), flip

    def _photometric(self, img, rng):
        img = img.astype(np.float32)
        img = img * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)            # contrast, brightness
        mean = img.mean()
        img = (img - mean) * rng.uniform(0.8, 1.2) + mean
        if rng.random() < 0.2:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
        if rng.random() < 0.2:
            img = img + rng.normal(0, rng.uniform(2, 8), img.shape)
        return np.clip(img, 0, 255)

    def __getitem__(self, i):
        r = self.recs[i]
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        img = cv2.imread(str(self.root / "crops" / f"{r['image']}.png"), cv2.IMREAD_COLOR)
        M_aug, flip = self._ctx_to_out(rng)
        img = cv2.warpAffine(img, M_aug, (self.out, self.out), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        if self.train:
            img = self._photometric(img, rng)
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD

        kp = G.apply_affine(M_aug, np.array(r["kp_ctx"]))
        exists = np.array(r["exists"], bool)
        in_image = np.array(r["in_image"], bool)
        if flip:
            kp, exists, in_image = kp[self.flip_idx], exists[self.flip_idx], in_image[self.flip_idx]
        in_crop = (kp[:, 0] >= 0) & (kp[:, 0] <= self.out - 1) & (kp[:, 1] >= 0) & (kp[:, 1] <= self.out - 1)
        # COCO-style: 0 no physical point, 1 exists but outside crop/image, 2 visible
        v = np.where(exists, np.where(in_crop & in_image, 2, 1), 0).astype(np.int64)

        hm_xy = (kp + 0.5) / self.stride - 0.5   # pixel-centre convention
        xs, ys = self.grid
        d2 = (xs[None] - hm_xy[:, 0, None, None]) ** 2 + (ys[None] - hm_xy[:, 1, None, None]) ** 2
        target = np.exp(-d2 / (2 * self.sigma ** 2)).astype(np.float32)
        target[v != 2] = 0.0

        M_native_to_out = G.compose(M_aug, np.array(r["M_ctx"]))
        return {
            "img": torch.from_numpy(img.transpose(2, 0, 1).copy()),
            "target": torch.from_numpy(target),
            "v": torch.from_numpy(v),
            "kp": torch.from_numpy(kp.astype(np.float32)),
            "M": torch.from_numpy(M_native_to_out.astype(np.float64)),
            "rho": torch.tensor(r["physics"]["r_inner_m"] / r["physics"]["R_m"], dtype=torch.float32),
            "flip": flip,
            "idx": i,
        }
