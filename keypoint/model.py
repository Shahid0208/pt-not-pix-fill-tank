"""ResNet-18 SimpleBaseline (Xiao et al. 2018) with a stride-2 heatmap head and a validity head."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class KeypointNet(nn.Module):
    def __init__(self, num_kp=65, deconv_layers=4, deconv_ch=256, pretrained=True, vis_detach=False):
        super().__init__()
        self.vis_detach = vis_detach  # validity head gets no gradient path into the backbone
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        layers, cin = [], 512
        for _ in range(deconv_layers):
            layers += [nn.ConvTranspose2d(cin, deconv_ch, 4, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(deconv_ch), nn.ReLU(inplace=True)]
            cin = deconv_ch
        self.deconv = nn.Sequential(*layers)
        self.final = nn.Conv2d(deconv_ch, num_kp, 1)
        self.validity = nn.Linear(512, num_kp)
        for m in self.deconv.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
        nn.init.normal_(self.final.weight, std=0.001)
        nn.init.zeros_(self.final.bias)

    def forward(self, x):
        f = self.backbone(x)
        hm = self.final(self.deconv(f))
        pooled = F.adaptive_avg_pool2d(f.detach() if self.vis_detach else f, 1).flatten(1)
        val = self.validity(pooled)
        return hm, val


# Defaults give the plain heatmap loss: L = L_hm + w_vis * L_vis.
LOSS_DEFAULTS = {
    "w_vis": 0.1,
    "hm_scale": 1.0,          # multiplies L_hm (per-pixel MSE is tiny: predicting all zeros scores ~7.7e-4)
    "group_weights": None,    # {"center": 1, "rim": 1, "inner_shadow": 3, ...}; None = all 1
    "w_coord": 0.0,           # soft-argmax coordinate loss (smooth L1, input px)
    "w_disp": 0.0,            # physics displacement loss: inner - rim_inner and ground - rim (input px)
    "coord_window": 6,        # half-size of the soft-argmax window (heatmap px); 6 = 3 sigma, bias < 0.01 px
    "aux_start_epoch": 1,     # L_coord / L_disp off before this epoch, then ramped in over aux_ramp_epochs
    "aux_ramp_epochs": 0,
    "aux_hm_gate": None,      # if set: L_coord / L_disp stay off until train L_hm < this (heatmaps have peaks)
}
GROUP_INDEX = [("center", 0, 1), ("rim", 1, 17), ("inner_shadow", 17, 33), ("ground_shadow", 33, 49),
               ("roof_edge", 49, 65)]


def loss_cfg(cfg):
    return {k: cfg.get(k, v) for k, v in LOSS_DEFAULTS.items()}


def kp_weights(group_weights, num_kp, device):
    w = torch.ones(num_kp, device=device)
    if group_weights:
        for g, a, b in GROUP_INDEX:
            if b <= num_kp:
                w[a:b] = float(group_weights.get(g, 1.0))
    return w


def soft_argmax_local(hm, r, eps=1e-4):
    """Differentiable sub-pixel location: weighted mean of ReLU(heatmap) + eps in a (2r+1)^2 window
    centred on the heatmap's OWN argmax. hm (B, K, H, W) -> coords (B, K, 2) in heatmap px.

    - The window follows the prediction, never the true keypoint, so it leaks no target information.
      (A window on the true keypoint let a flat heatmap score as if it were correct.)
    - eps gives every window a small uniform mass: an empty or negative heatmap does not drop out of
      the loss; its location is simply the argmax.
    - r = 6 (3 sigma for sigma = 2) keeps the truncation bias below 0.01 heatmap px.
    """
    B, K, H, W = hm.shape
    with torch.no_grad():
        idx = hm.flatten(2).argmax(-1)
        cy, cx = idx // W, idx % W
    hp = F.pad(hm, (r, r, r, r))
    d = torch.arange(-r, r + 1, device=hm.device)
    flat = ((cy[..., None, None] + r + d[:, None]) * (W + 2 * r) + (cx[..., None, None] + r + d[None, :])).flatten(2)
    win = hp.flatten(2).gather(2, flat).clamp(min=0) + eps
    gy = (cy[..., None, None] + d[:, None]).expand(-1, -1, -1, 2 * r + 1).flatten(2).float()
    gx = (cx[..., None, None] + d[None, :]).expand(-1, -1, 2 * r + 1, -1).flatten(2).float()
    mass = win.sum(-1)
    return torch.stack([(win * gx).sum(-1) / mass, (win * gy).sum(-1) / mass], -1)


def keypoint_loss(hm, val, target, v, cfg, kp=None, rho=None, stride=2.0, aux_scale=1.0):
    """Total loss and its parts.

    L = hm_scale * L_hm + w_vis * L_vis + aux_scale * (w_coord * L_coord + w_disp * L_disp)
      L_hm    per-pixel MSE to the Gaussian targets, averaged over visible keypoints (group-weighted)
      L_vis   BCE: is the keypoint visible in the crop
      L_coord smooth-L1 between the predicted location (soft-argmax around the predicted peak) and the
              true keypoint, input px, group-weighted
      L_disp  smooth-L1 between predicted and true displacements, the quantities the physics uses:
              inner_k - rim_inner_k (rim_inner = c + rho (rim - c), rho = r_inner / R) and ground_k - rim_k
    aux_scale ramps L_coord / L_disp in after the heatmaps have formed peaks (see aux_ramp in train.py).
    Parts are logged unweighted.
    """
    c = loss_cfg(cfg)
    hm = hm.float()
    vis = (v == 2).float()
    w = kp_weights(c["group_weights"], hm.shape[1], hm.device)[None] * vis          # (B, K)
    per_kp = ((hm - target) ** 2).mean(dim=(2, 3))
    l_hm = (per_kp * w).sum() / w.sum().clamp(min=1e-6)
    l_vis = F.binary_cross_entropy_with_logits(val.float(), vis)
    total = c["hm_scale"] * l_hm + c["w_vis"] * l_vis
    parts = {"loss_hm": l_hm, "loss_vis": l_vis}
    if (c["w_coord"] > 0 or c["w_disp"] > 0) and kp is not None:
        xy = (soft_argmax_local(hm, int(c["coord_window"])) + 0.5) * stride - 0.5    # -> input px
        kp = kp.float()
        if c["w_coord"] > 0:
            err = F.smooth_l1_loss(xy, kp, reduction="none", beta=1.0).sum(-1)
            l_coord = (err * w).sum() / w.sum().clamp(min=1e-6)
            total = total + aux_scale * c["w_coord"] * l_coord
            parts["loss_coord"] = l_coord
        if c["w_disp"] > 0 and rho is not None and hm.shape[1] >= 49:
            rho = rho.float()[:, None, None]

            def disp(p):
                cen, rim = p[:, 0:1], p[:, 1:17]
                return p[:, 17:33] - (cen + rho * (rim - cen)), p[:, 33:49] - rim
            pi, pg = disp(xy)
            ti, tg = disp(kp)
            mi = vis[:, 17:33] * vis[:, 1:17] * vis[:, 0:1]
            mg = vis[:, 33:49] * vis[:, 1:17]
            ei = F.smooth_l1_loss(pi, ti, reduction="none", beta=1.0).sum(-1)
            eg = F.smooth_l1_loss(pg, tg, reduction="none", beta=1.0).sum(-1)
            l_disp = ((ei * mi).sum() + (eg * mg).sum()) / (mi.sum() + mg.sum()).clamp(min=1e-6)
            total = total + aux_scale * c["w_disp"] * l_disp
            parts["loss_disp"] = l_disp
    parts["loss"] = total
    return total, parts


def decode_heatmaps(hm, stride, kernel=11):
    """Argmax + DARK Taylor refinement (Zhang et al. 2020). hm: (B, K, H, W) numpy -> (B, K, 2) input px."""
    B, K, H, W = hm.shape
    flat = hm.reshape(B, K, -1)
    idx = flat.argmax(-1)
    xs, ys = (idx % W).astype(np.float64), (idx // W).astype(np.float64)
    import cv2
    out = np.stack([xs, ys], -1)
    for b in range(B):
        for k in range(K):
            h = cv2.GaussianBlur(hm[b, k].astype(np.float32), (kernel, kernel), 0)
            h = np.log(np.maximum(h, 1e-10))
            x, y = int(xs[b, k]), int(ys[b, k])
            if 1 < x < W - 2 and 1 < y < H - 2:
                dx = 0.5 * (h[y, x + 1] - h[y, x - 1])
                dy = 0.5 * (h[y + 1, x] - h[y - 1, x])
                dxx = 0.25 * (h[y, x + 2] - 2 * h[y, x] + h[y, x - 2])
                dyy = 0.25 * (h[y + 2, x] - 2 * h[y, x] + h[y - 2, x])
                dxy = 0.25 * (h[y + 1, x + 1] - h[y - 1, x + 1] - h[y + 1, x - 1] + h[y - 1, x - 1])
                det = dxx * dyy - dxy ** 2
                if det > 1e-12:
                    off = -np.array([dyy * dx - dxy * dy, dxx * dy - dxy * dx]) / det
                    out[b, k] += np.clip(off, -1.0, 1.0)
    return (out + 0.5) * stride - 0.5
