"""Train a YOLO-OBB tank detector with Ultralytics, track train/val losses, then score on the test split.

    python yolo_obb/train.py --config yolo_obb/config.yaml
    python yolo_obb/train.py --config yolo_obb/config.yaml data=yolo_data/data_real.yaml name=real_yolo26s
    python yolo_obb/train.py --resume runs/obb/mixed_yolo26s/weights/last.pt

Outputs (all in <project>/<name>/), the graphs redrawn after every epoch:
    results.csv            per epoch, REAL val: train/{box,cls,l1,angle}_loss, val/{...}_loss, precision,
                           recall, mAP50, mAP50-95, lr (Ultralytics)
    results_synthetic.csv  per epoch, SYNTHETIC val (synthetic/val.txt next to the data YAML): the same
                           val losses and metrics, from a second validation of the same EMA model
    losses.png             train vs real val, every loss + real metrics
    losses_synthetic.png   train vs synthetic val, every loss + synthetic metrics
    losses_comparison.png  real val vs synthetic val, every loss and metric
    results.png, confusion matrices, PR curves, val_batch*_pred.jpg, events.* (TensorBoard)
    final_eval/            best.pt on real test and synthetic test: Ultralytics plots for each,
                           metrics.csv, comparison.png (compare_real_synth.py)
"""
import argparse
import os
import csv
from copy import copy
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO, settings

from plot_losses import plot_run

NON_TRAIN_KEYS = ("model", "data", "sensor_aug")


def sensor_augmentations():
    """Pixel-only Albumentations transforms: Ultralytics' default extras (Blur, MedianBlur, ToGray,
    CLAHE) plus effects simulating NAIP sensor and quality variation (blur, noise, JPEG, coarser
    resolution). They never move pixels, so boxes stay exact. Passed as Ultralytics' `augmentations`
    argument, which replaces its default list, so the defaults are repeated here. They are stored in
    checkpoints, so a resumed run keeps them."""
    import albumentations as A
    import cv2

    return [
        # Ultralytics' default extras (kept; CLAHE raised from 1% to 5% below)
        A.Blur(blur_limit=(3, 7), p=0.01),
        A.MedianBlur(blur_limit=(3, 7), p=0.01),
        # sensor / quality variation
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.3, 1.2), p=0.10),
        A.GaussNoise(std_range=(0.01, 0.04), p=0.10),
        A.ImageCompression(quality_range=(60, 95), p=0.10),
        # area down / linear up looks like coarser imagery; the default (nearest) adds blocky artefacts
        A.Downscale(scale_range=(0.5, 0.85),
                    interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}, p=0.10),
        A.CLAHE(clip_limit=(1.0, 3.0), p=0.05),
        A.ToGray(p=0.01),
    ]


def parse_overrides(items):
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        if not _:
            raise SystemExit(f"override must be key=value, got {item!r}")
        out[key] = yaml.safe_load(value)
    return out


def resolve_data_yaml(data_path: Path) -> str:
    """Point the dataset `path` at the YAML's own folder so a copied dataset (e.g. to a local disk) just works."""
    data = yaml.safe_load(data_path.read_text())
    data["path"] = str(data_path.resolve().parent)
    resolved = data_path.with_name(f".resolved_{data_path.name}")
    resolved.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(resolved)


def log_epoch(trainer):
    """One line per epoch in the job log, so losses can be followed with `tail -f` during a Slurm run."""
    if trainer.epoch >= trainer.epochs:  # Ultralytics fires this again for the final best.pt validation
        return
    train = trainer.label_loss_items(trainer.tloss, prefix="train")
    m = trainer.metrics
    parts = [f"epoch {trainer.epoch + 1}/{trainer.epochs}"]
    parts += [f"{k}={v:.4f}" for k, v in train.items()]
    parts += [f"{k}={v:.4f}" for k, v in m.items() if k.startswith("val/")]
    parts += [f"mAP50={m.get('metrics/mAP50(B)', 0):.4f}", f"mAP50-95={m.get('metrics/mAP50-95(B)', 0):.4f}"]
    print(" | ".join(parts), flush=True)


class SyntheticVal:
    """After every epoch, validate the same EMA model on the synthetic val split as well, and append its
    val losses and metrics to results_synthetic.csv. Model selection (best.pt) still uses real val only.
    Any error turns this off with a warning; it never stops training."""

    LIST = "synthetic/val.txt"  # relative to the dataset root (the data YAML's `path`)

    def __init__(self):
        self.validator, self.off = None, False

    def __call__(self, trainer):
        if self.off or trainer.epoch >= trainer.epochs:
            return
        lst = Path(trainer.data["path"]) / self.LIST
        if not lst.exists():
            print(f"NOTE: no {lst}; skipping per-epoch synthetic validation", flush=True)
            self.off = True
            return
        try:
            if self.validator is None:
                from ultralytics.models.yolo.obb import OBBValidator

                loader = trainer.get_dataloader(str(lst), trainer.batch_size, rank=-1, mode="val")
                args = copy(trainer.args)
                args.plots = False
                self.validator = OBBValidator(loader, save_dir=Path(trainer.save_dir) / "synthetic_val", args=args)
            metrics = self.validator(trainer=trainer)
            epoch = trainer.epoch + 1
            write_synthetic_row(Path(trainer.save_dir) / "results_synthetic.csv", epoch, metrics)
            parts = [f"synthetic {epoch}/{trainer.epochs}"]
            parts += [f"{k}={v:.4f}" for k, v in metrics.items() if k.startswith("val/")]
            parts += [f"mAP50={metrics.get('metrics/mAP50(B)', 0):.4f}",
                      f"mAP50-95={metrics.get('metrics/mAP50-95(B)', 0):.4f}"]
            print(" | ".join(parts), flush=True)
        except Exception as e:
            self.off = True
            print(f"WARNING: synthetic validation turned off: {e}", flush=True)


def write_synthetic_row(path: Path, epoch: int, metrics: dict):
    """Write one row per epoch; rows from a later epoch (left by an interrupted run) are replaced."""
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(float(r["epoch"])) < epoch]
    rows.append({"epoch": epoch, **{k: f"{v:.5g}" for k, v in metrics.items()}})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[-1]))
        writer.writeheader()
        writer.writerows(rows)


def plot_epoch(trainer):
    """Redraw losses.png, losses_synthetic.png and losses_comparison.png after every epoch, so the
    graphs grow while training runs. Ultralytics writes results.csv just before this callback fires."""
    if trainer.epoch >= trainer.epochs:
        return
    try:
        plot_run(Path(trainer.save_dir), verbose=False)
    except Exception as e:  # a plotting problem must never stop training
        print(f"WARNING: could not update the loss graphs: {e}", flush=True)


RANK = int(os.environ.get("RANK", -1))   # set by torchrun when train.py itself is launched on several GPUs


def add_callbacks(model):
    """Logging, synthetic val and plots: in the main process only (rank -1 single GPU, rank 0 under torchrun).
    Launching train.py with torchrun (instead of Ultralytics' own multi-GPU relaunch) keeps these callbacks and
    the sensor augmentations, which Ultralytics' relaunch would drop."""
    if RANK not in (-1, 0):
        return
    model.add_callback("on_fit_epoch_end", log_epoch)
    if RANK == -1:
        # Per-epoch synthetic val only on a single GPU: under torchrun the validator synchronises all ranks, and
        # running it on rank 0 alone deadlocks (stuck after epoch 1 with all GPUs spinning).
        # The synthetic TEST is still scored at the end, after the process group is closed.
        model.add_callback("on_fit_epoch_end", SyntheticVal())
    model.add_callback("on_fit_epoch_end", plot_epoch)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--resume", type=Path, help="last.pt of an interrupted run; continues it with its own settings")
    ap.add_argument("overrides", nargs="*", help="key=value overrides of the config")
    args = ap.parse_args()

    settings.update({"tensorboard": True})

    if args.resume:
        # A resumed Slurm job may run on another node, where the dataset was re-staged;
        # data=... re-points the run at it (Ultralytics uses it when the saved path is gone).
        extra = parse_overrides(args.overrides)
        if "data" in extra:
            extra["data"] = resolve_data_yaml(Path(extra["data"]))
        model = YOLO(str(args.resume))
        add_callbacks(model)
        model.train(resume=True, **extra)
        cfg = {"data": model.trainer.args.data, "imgsz": model.trainer.args.imgsz,
               "batch": model.trainer.args.batch, "device": model.trainer.args.device}
    else:
        cfg = yaml.safe_load(args.config.read_text()) | parse_overrides(args.overrides)
        cfg["data"] = resolve_data_yaml(Path(cfg["data"]))
        # Ultralytics nests a relative project under its own runs/<task>/ folder; make it absolute
        # so `project: runs/obb` means <cwd>/runs/obb.
        cfg["project"] = str(Path(cfg["project"]).resolve())
        model = YOLO(cfg["model"])
        add_callbacks(model)
        extra = {"augmentations": sensor_augmentations()} if cfg.get("sensor_aug") else {}
        model.train(data=cfg["data"], **{k: v for k, v in cfg.items() if k not in NON_TRAIN_KEYS}, **extra)

    if RANK not in (-1, 0):          # other torchrun ranks: training done; rank 0 writes plots and test scores
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    save_dir = Path(model.trainer.save_dir)
    plot_run(save_dir)
    if RANK == 0:
        # Ultralytics read RANK=0 at import, so its validator would gather over the closed process group and crash.
        # Score best.pt in a separate process instead: compare_real_synth.py.
        print(f"torchrun: final test evaluation skipped; run compare_real_synth.py --model {save_dir}/weights/best.pt")
        return

    # Final scores of best.pt on the test splits (never seen in training or model selection):
    # real test and synthetic test, each with full plots, plus the real-vs-synthetic comparison.
    from compare_real_synth import run_comparison

    root = Path(model.trainer.data["path"])
    sets = {name: lst for name, lst in (("real_test", "real/test.txt"), ("synthetic_test", "synthetic/test.txt"))
            if (root / lst).exists()}
    results = run_comparison(save_dir / "weights" / "best.pt", root, sets, save_dir / "final_eval",
                             imgsz=cfg["imgsz"], batch=cfg["batch"],
                             device=str(cfg["device"]).split(",")[0])   # final test on one GPU
    for name, r in results.items():
        print(f"TEST {name}  mAP50={r['overall'][2]:.4f}  mAP50-95={r['overall'][3]:.4f}")


if __name__ == "__main__":
    main()
