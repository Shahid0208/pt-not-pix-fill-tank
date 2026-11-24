"""Evaluate a trained model separately on real and synthetic data and compare the two.

    python yolo_obb/compare_real_synth.py --model runs/obb/ft20/weights/best.pt --root yolo_data \
        --out runs/obb/ft20/final_eval                  # default: real test vs synthetic test
    # any two lists, e.g. the validation splits:
    python yolo_obb/compare_real_synth.py --model ... --real-list real/val.txt --real-name real_val \
        --synth-list synthetic/val.txt --synth-name synthetic_val --out ...

train.py runs this automatically on real test vs synthetic test at the end of training.

Writes to --out:
    <real-name>/, <synth-name>/   Ultralytics val plots on each set (PR/F1/P/R curves, confusion
                                  matrices, predictions drawn on images)
    metrics.csv                   overall and per-class precision, recall, mAP50, mAP50-95, box counts
    comparison.png                real vs synthetic: scores, PR curves, box sizes, tanks per image, box
                                  angles, image brightness and saturation
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from common import CLASS_NAMES  # noqa: E402

PAD_VALUE = 114  # grey padding prepare_synthetic.py puts around small rescaled renders
IMG_SIZE = 512  # real chips and synthetic tiles are all 512x512; labels are normalised


def image_paths(root: Path, list_file: str):
    lst = root / list_file
    return [lst.parent / line.strip()[2:] for line in lst.read_text().splitlines() if line.strip()]


def label_stats(images):
    """Box size (sqrt of area, px), angle (deg, 0-90), class, and boxes per image from YOLO-OBB labels."""
    sizes, angles, classes, per_image = [], [], [], []
    for img in images:
        lbl = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
        rows = [r.split() for r in lbl.read_text().splitlines() if r.strip()] if lbl.exists() else []
        per_image.append(len(rows))
        for r in rows:
            p = np.array(r[1:], float).reshape(4, 2) * IMG_SIZE
            e1, e2 = p[1] - p[0], p[2] - p[1]
            sizes.append(np.sqrt(np.linalg.norm(e1) * np.linalg.norm(e2)))
            angles.append(np.degrees(np.arctan2(e1[1], e1[0])) % 90)
            classes.append(int(r[0]))
    return np.array(sizes), np.array(angles), np.array(classes), np.array(per_image)


def image_stats(images, n=600, seed=0):
    """Mean brightness (HSV V) and saturation (HSV S) of a random sample of images. The grey padding
    around rescaled synthetic renders (114, 114, 114; JPEG-blurred) is left out, so only scene
    pixels are compared."""
    sample = random.Random(seed).sample(images, min(n, len(images)))
    v, s = [], []
    for p in sample:
        bgr = cv2.imread(str(p))
        scene = np.abs(bgr.astype(int) - PAD_VALUE).max(axis=2) > 3
        if scene.sum() < 100:
            continue
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s.append(hsv[..., 1][scene].mean())
        v.append(hsv[..., 2][scene].mean())
    return np.array(v), np.array(s)


def label_of(key: str) -> str:
    return key.replace("_", " ")  # e.g. "real_test" -> "real test"


def evaluate(model, root: Path, key: str, list_file: str, out: Path, imgsz, batch, device):
    data = {"path": str(root.resolve()), "train": list_file, "val": list_file,
            "names": dict(enumerate(CLASS_NAMES))}
    data_yaml = out / f"data_{key}.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False))
    m = model.val(data=str(data_yaml), split="val", imgsz=imgsz, batch=batch, device=device,
                  project=str(out), name=key, exist_ok=True, plots=True)
    box = m.box
    per_class = {CLASS_NAMES[c]: (box.p[j], box.r[j], box.ap50[j], box.ap[j])
                 for j, c in enumerate(box.ap_class_index)}
    pr = m.curves_results[0]  # [recall x, precision y (per class), xlabel, ylabel]
    pr_curves = {CLASS_NAMES[c]: np.asarray(pr[1])[j] for j, c in enumerate(box.ap_class_index)}
    return {"overall": (box.mp, box.mr, box.map50, box.map), "per_class": per_class,
            "pr_x": np.asarray(pr[0]), "pr": pr_curves}


def write_metrics(results, stats, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "class", "boxes", "precision", "recall", "mAP50", "mAP50-95"])
        for key, r in results.items():
            classes = stats[key][2]
            w.writerow([key, "all", len(classes), *[f"{v:.4f}" for v in r["overall"]]])
            for name, vals in r["per_class"].items():
                n = int((classes == CLASS_NAMES.index(name)).sum())
                w.writerow([key, name, n, *[f"{v:.4f}" for v in vals]])


def plot(results, stats, img_stats, path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    keys = list(results)
    COLORS = {k: f"C{i}" for i, k in enumerate(keys)}
    LABELS = {k: label_of(k) for k in keys}

    # 1. overall scores
    ax = axes[0, 0]
    names = ["precision", "recall", "mAP50", "mAP50-95"]
    x = np.arange(len(names))
    for i, k in enumerate(keys):
        vals = results[k]["overall"]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=COLORS[k], label=LABELS[k])
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.2)
    ax.set_title("Overall scores (all classes present)")
    ax.legend(fontsize=8, loc="upper center", ncol=2)

    # 2. per-class mAP50-95 for classes present in both sets
    ax = axes[0, 1]
    shared = [c for c in CLASS_NAMES if all(c in results[k]["per_class"] for k in keys)]
    x = np.arange(len(shared))
    for i, k in enumerate(keys):
        vals = [results[k]["per_class"][c][3] for c in shared]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=COLORS[k], label=LABELS[k])
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.set_xticks(x, [c.replace("_tank", "").replace("_", "\n") for c in shared], fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_title("mAP50-95 per class (classes in both sets)")
    ax.legend(fontsize=8, loc="upper center", ncol=2)

    # 3. PR curves of the shared classes
    ax = axes[0, 2]
    for k in keys:
        for c, ls in zip(shared, ["-", "--", ":", "-."]):
            ax.plot(results[k]["pr_x"], results[k]["pr"][c], color=COLORS[k], ls=ls, lw=1.2,
                    label=f"{LABELS[k]}: {c}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curves")
    ax.legend(fontsize=6)

    # 4-9. data statistics
    def hist(ax, values, bins, title, xlabel, log=False):
        for k in keys:
            ax.hist(values[k], bins=bins, color=COLORS[k], alpha=0.55, density=True,
                    label=f"{LABELS[k]} (n={len(values[k])})")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        if log:
            ax.set_xscale("log")
        ax.legend(fontsize=7)

    hist(axes[1, 0], {k: stats[k][0] for k in keys}, np.logspace(0.3, 2.6, 40),
         "Box size (all classes)", "sqrt(box area) [px]", log=True)
    for ax, cname in zip([axes[1, 1], axes[1, 2]], ["external_floating_roof_tank", "closed_roof_tank"]):
        cid = CLASS_NAMES.index(cname)
        hist(ax, {k: stats[k][0][stats[k][2] == cid] for k in keys}, np.logspace(0.3, 2.6, 40),
             f"Box size: {cname}", "sqrt(box area) [px]", log=True)
    hist(axes[2, 0], {k: stats[k][3] for k in keys}, np.arange(0, 41) - 0.5,
         "Tanks per image", "boxes per image")
    hist(axes[2, 1], {k: stats[k][1] for k in keys}, np.linspace(0, 90, 31),
         "Box angle (0 = axis-aligned)", "degrees")
    ax = axes[2, 2]
    for k in keys:
        v, s = img_stats[k]
        ax.scatter(v, s, s=5, alpha=0.4, color=COLORS[k], label=LABELS[k])
    ax.set_title("Image appearance (random sample, padding excluded)")
    ax.set_xlabel("mean brightness (HSV V)")
    ax.set_ylabel("mean saturation (HSV S)")
    ax.legend(fontsize=7, markerscale=3)

    for a in axes.flat:
        a.grid(alpha=0.3)
    note = " Training-set data measures fit, not generalisation." if any("train" in k for k in keys) else ""
    fig.suptitle(f"{LABELS[keys[0]]} vs {LABELS[keys[1]]}.{note}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_comparison(model_path, root: Path, sets: dict, out: Path, imgsz=None, batch=16, device="0"):
    """Evaluate `model_path` on each {name: list file relative to root}; write plots, metrics.csv and
    comparison.png to `out`. imgsz=None uses the size the model was trained at."""
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    imgsz = imgsz or model.overrides.get("imgsz", 512)
    results, stats, img_stats = {}, {}, {}
    for key, list_file in sets.items():
        images = image_paths(root, list_file)
        print(f"\n=== {key}: {len(images)} images (imgsz {imgsz}) ===", flush=True)
        results[key] = evaluate(model, root, key, list_file, out, imgsz, batch, device)
        stats[key] = label_stats(images)
        img_stats[key] = image_stats(images)
    write_metrics(results, stats, out / "metrics.csv")
    if len(results) > 1:
        plot(results, stats, img_stats, out / "comparison.png")
    print(f"\nsaved {out / 'metrics.csv'}" + (f" and {out / 'comparison.png'}" if len(results) > 1 else ""))
    for key, r in results.items():
        p, rc, m50, m = r["overall"]
        print(f"{key:16s} P={p:.3f} R={rc:.3f} mAP50={m50:.3f} mAP50-95={m:.3f}")
        for name, (cp, cr, c50, c) in r["per_class"].items():
            print(f"  {name:30s} P={cp:.3f} R={cr:.3f} mAP50={c50:.4f}  mAP50-95={c:.4f}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("yolo_data"))
    ap.add_argument("--out", type=Path, default=Path("runs/obb/eval_real_vs_synth"))
    ap.add_argument("--imgsz", type=int, help="default: the size the model was trained at")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--real-list", default="real/test.txt", help="real image list, relative to --root")
    ap.add_argument("--real-name", default="real_test")
    ap.add_argument("--synth-list", default="synthetic/test.txt", help="synthetic image list, relative to --root")
    ap.add_argument("--synth-name", default="synthetic_test")
    args = ap.parse_args()
    run_comparison(args.model, args.root, {args.real_name: args.real_list, args.synth_name: args.synth_list},
                   args.out, args.imgsz, args.batch, args.device)


if __name__ == "__main__":
    main()
