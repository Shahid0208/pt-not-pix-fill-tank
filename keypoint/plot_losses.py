"""Loss curves from results.csv: one panel per loss term, train (solid) vs val (dashed).

    python keypoint/plot_losses.py runs/keypoint/kp_r18_s0                  # -> <run>/losses.png
    python keypoint/plot_losses.py runs/keypoint/a runs/keypoint/b -o cmp.png  # compare runs (val only)
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRAIN, VAL = "#2a78d6", "#eb6834"   # categorical slots 1 and 2
RUNS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
PANELS = [("loss", "Total loss L"), ("loss_hm", "Heatmap loss L_hm"), ("loss_vis", "Validity loss L_vis")]
EXTRA = [("loss_coord", "Coordinate loss L_coord"), ("loss_disp", "Displacement loss L_disp")]


def read(run):
    with open(Path(run) / "results.csv") as f:
        return [{k: float(v) if v not in ("", None) else float("nan") for k, v in r.items()}
                for r in csv.DictReader(f)]


def _style(ax, title, ylabel):
    ax.set_title(title, fontsize=10, loc="left", color="#222")
    ax.set_xlabel("epoch", fontsize=9, color="#555")
    ax.set_ylabel(ylabel, fontsize=9, color="#555")
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbb")
    ax.tick_params(colors="#555", labelsize=8)


def plot_run(run, out=None):
    rows = read(run)
    ep = [r["epoch"] for r in rows]
    panels = PANELS + [p for p in EXTRA if any((r.get(f"train_{p[0]}") or 0) > 0 for r in rows)]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(3.75 * (len(panels) + 1), 3.4))
    for ax, (key, title) in zip(axes, panels):
        ax.plot(ep, [r[f"train_{key}"] for r in rows], color=TRAIN, lw=2, marker="o", ms=4, label="train")
        ax.plot(ep, [r[f"val_{key}"] for r in rows], color=VAL, lw=2, ls="--", marker="s", ms=4, label="val")
        _style(ax, title, "loss")
        ax.legend(fontsize=8, frameon=False)
    ax = axes[-1]
    ax.plot(ep, [r["val_px_err"] for r in rows], color=VAL, lw=2, ls="--", marker="s", ms=4, label="val")
    _style(ax, "Val keypoint error (native px)", "mean px error")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out or Path(run) / "losses.png", dpi=120)
    plt.close(fig)


def compare(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for i, run in enumerate(runs):
        rows = read(run)
        ep = [r["epoch"] for r in rows]
        for ax, (key, title) in zip(axes, PANELS):
            ax.plot(ep, [r[f"val_{key}"] for r in rows], color=RUNS[i % len(RUNS)], lw=2, label=Path(run).name)
            _style(ax, f"Val {title}", "loss")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    if len(a.runs) == 1:
        plot_run(a.runs[0], a.out)
    else:
        compare(a.runs, a.out or "compare.png")
