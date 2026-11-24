"""Plot train vs val loss per term and the val metrics from one or more Ultralytics runs.

    python yolo_obb/plot_losses.py runs/obb/mixed_yolo26s                       # -> <run>/losses.png
    python yolo_obb/plot_losses.py runs/obb/real_yolo26s runs/obb/mixed_yolo26s -o compare.png

--chain joins runs that continue each other (e.g. 10 epochs + 5 more from its weights) into one
training: epochs are renumbered 1..N, `time` keeps adding up, and a merged results.csv is written
next to the graph.

    python yolo_obb/plot_losses.py --chain runs/obb/ft10_yolo26s runs/obb/ft10_plus5 \
        -o runs/obb/ft15_merged/losses.png       # -> ft15_merged/losses.png + results.csv (epochs 1-15)

--all draws every graph of one run (train.py does this after every epoch):
    losses.png             train vs real val, every loss + real metrics          (results.csv)
    losses_synthetic.png   train vs synthetic val, every loss + synthetic metrics (results_synthetic.csv)
    losses_comparison.png  real val vs synthetic val, every loss and metric
    python yolo_obb/plot_losses.py --all runs/obb/ft20
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

METRICS = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]


def load(run: Path) -> dict:
    """results.csv -> {column: [float, ...]} (Ultralytics pads some column names with spaces)."""
    with open(run / "results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return {k.strip(): [float(r[k]) for r in rows] for k in rows[0]}


def chain(runs):
    """Concatenate consecutive runs: epochs and cumulative time continue from the previous run.
    Returns the merged table and the epochs after which each later run starts."""
    tables = [load(run) for run in runs]
    columns = [c for c in tables[0] if all(c in t for t in tables)]
    merged = {c: [] for c in columns}
    boundaries, epoch_offset, time_offset = [], 0.0, 0.0
    for i, t in enumerate(tables):
        if i:
            boundaries.append(epoch_offset)
        for c in columns:
            shift = epoch_offset if c == "epoch" else time_offset if c == "time" else 0.0
            merged[c] += [v + shift for v in t[c]]
        epoch_offset, time_offset = merged["epoch"][-1], merged.get("time", [0.0])[-1]
    return merged, boundaries


def write_csv(table: dict, path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(table)
        for row in zip(*table.values()):
            writer.writerow([int(v) if k == "epoch" else f"{v:.6g}" for k, v in zip(table, row)])


def plot_frames(frames: dict, out: Path, boundaries=(), verbose=True):
    first = next(iter(frames.values()))
    terms = [c.split("/", 1)[1] for c in first if c.startswith("train/")]
    panels = [("loss", t) for t in terms] + [("metric", m) for m in METRICS if m in first]

    ncols = 4
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax, (kind, key) in zip(axes.flat, panels):
        for i, (name, df) in enumerate(frames.items()):
            color = f"C{i}"
            label = f"{name} " if len(frames) > 1 else ""
            if kind == "loss":
                ax.plot(df["epoch"], df[f"train/{key}"], color=color, marker="o", ms=3, label=f"{label}train")
                if f"val/{key}" in df:
                    ax.plot(df["epoch"], df[f"val/{key}"], color=color, ls="--", marker="o", ms=3,
                            label=f"{label}val")
                ax.set_title(key)
            else:
                ax.plot(df["epoch"], df[key], color=color, marker="o", ms=3, label=name)
                ax.set_title(key.removeprefix("metrics/").removesuffix("(B)"))
        for b in boundaries:  # where a chained run continues from the previous one's weights
            ax.axvline(b + 0.5, color="grey", ls=":", lw=1)
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    if boundaries:
        starts = ", ".join(f"{int(b) + 1}" for b in boundaries)
        fig.suptitle(f"Dotted line: training continued from the previous run's weights (from epoch {starts})",
                     fontsize=9, color="grey")
    fig.tight_layout()
    # Write to a temp file and swap it in, so a viewer never opens a half-written image
    # (the graph is redrawn after every epoch while training runs).
    tmp = out.with_name(f".{out.stem}.tmp{out.suffix}")
    fig.savefig(tmp, dpi=120)
    plt.close(fig)
    os.replace(tmp, out)
    if verbose:
        print(f"saved {out}")


def plot_runs(runs, out: Path, verbose=True):
    plot_frames({run.name: load(run) for run in runs}, out, verbose=verbose)


def load_synthetic(run: Path, real: dict):
    """results_synthetic.csv (synthetic val losses + metrics per epoch) joined with the train losses of
    the same epochs from results.csv, so it plots exactly like results.csv. None if not there."""
    path = run / "results_synthetic.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    synth = {k.strip(): [float(r[k]) for r in rows] for k in rows[0]}
    row_of = {e: i for i, e in enumerate(real["epoch"])}
    keep = [i for i, e in enumerate(synth["epoch"]) if e in row_of]
    frame = {k: [v[i] for i in keep] for k, v in synth.items()}
    for k in real:
        if k.startswith("train/"):
            frame[k] = [real[k][row_of[e]] for e in frame["epoch"]]
    return frame


def plot_comparison(real: dict, synth: dict, out: Path, verbose=True):
    """Real val vs synthetic val per epoch: every loss term (with the train loss for reference) and
    every metric."""
    terms = [c.split("/", 1)[1] for c in real if c.startswith("train/")]
    panels = [("loss", t) for t in terms] + [("metric", m) for m in METRICS if m in real and m in synth]
    ncols = 4
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax, (kind, key) in zip(axes.flat, panels):
        if kind == "loss":
            ax.plot(real["epoch"], real[f"train/{key}"], color="grey", marker="o", ms=2, lw=1, label="train")
            for frame, color, name in ((real, "C0", "real val"), (synth, "C1", "synthetic val")):
                if f"val/{key}" in frame:
                    ax.plot(frame["epoch"], frame[f"val/{key}"], color=color, ls="--", marker="o", ms=3, label=name)
            ax.set_title(key)
        else:
            ax.plot(real["epoch"], real[key], color="C0", marker="o", ms=3, label="real val")
            ax.plot(synth["epoch"], synth[key], color="C1", marker="o", ms=3, label="synthetic val")
            ax.set_title(key.removeprefix("metrics/").removesuffix("(B)"))
        ax.set_xlabel("epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.suptitle("Real val vs synthetic val, per epoch", fontsize=10)
    fig.tight_layout()
    tmp = out.with_name(f".{out.stem}.tmp{out.suffix}")
    fig.savefig(tmp, dpi=120)
    plt.close(fig)
    os.replace(tmp, out)
    if verbose:
        print(f"saved {out}")


def plot_run(run: Path, verbose=True):
    """All graphs of one run folder: real, and synthetic + comparison when results_synthetic.csv exists."""
    real = load(run)
    plot_frames({run.name: real}, run / "losses.png", verbose=verbose)
    synth = load_synthetic(run, real)
    if synth:
        plot_frames({f"{run.name} (synthetic val)": synth}, run / "losses_synthetic.png", verbose=verbose)
        plot_comparison(real, synth, run / "losses_comparison.png", verbose=verbose)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="Ultralytics run folders (containing results.csv)")
    ap.add_argument("-o", "--out", type=Path, help="output image (default: <first run>/losses.png)")
    ap.add_argument("--all", action="store_true", help="draw losses.png, losses_synthetic.png and "
                    "losses_comparison.png for each run folder")
    ap.add_argument("--chain", action="store_true",
                    help="runs continue each other: merge them into one epoch 1..N curve and results.csv")
    args = ap.parse_args()

    if args.all:
        for run in args.runs:
            plot_run(run)
    elif args.chain:
        if not args.out:
            raise SystemExit("--chain needs -o, e.g. -o runs/obb/ft15_merged/losses.png")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        merged, boundaries = chain(args.runs)
        write_csv(merged, args.out.parent / "results.csv")
        print(f"saved {args.out.parent / 'results.csv'} (epochs 1-{int(merged['epoch'][-1])})")
        name = " + ".join(run.name for run in args.runs)
        plot_frames({name: merged}, args.out, boundaries)
    else:
        plot_runs(args.runs, args.out or args.runs[0] / "losses.png")


if __name__ == "__main__":
    main()
