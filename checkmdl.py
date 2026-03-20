#!/usr/bin/env python3
"""
checkmdl.py – Visualise a trained EMGSpeech model checkpoint.

Usage:
    python checkmdl.py <model_path> [output_dir]

Arguments:
    model_path : path to a saved PyTorch state-dict (.pt file),
                 e.g.  output/model.pt  or  snn_output/snn_model.pt
    output_dir : (optional) directory where all output files are written.
                 Defaults to the same directory as model_path.

Output files:
    architecture.txt          – plain-text layer summary
    summary_dashboard.png     – overview: pie chart, global histogram, std-dev
                                bar, layer statistics table
    parameter_counts.png      – horizontal bar chart of params per module
    weight_distributions.png  – per-layer weight histograms
    weight_heatmaps.png       – diverging heatmaps for 2-D weight matrices
    weight_norms.png          – L2 norm & parameter count per layer
"""

import os
import sys
from collections import OrderedDict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless / no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_state_dict(path: str) -> "OrderedDict[str, torch.Tensor]":
    """Load model state-dict from *path*.

    Handles both ``torch.save(model.state_dict(), ...)`` and
    ``torch.save(model, ...)`` checkpoints.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        # Could be a plain state-dict or a training checkpoint that wraps one.
        if "model_state_dict" in obj:
            return obj["model_state_dict"]
        if "state_dict" in obj:
            return obj["state_dict"]
        return obj  # assume it is already a state-dict
    # Full model saved with torch.save(model, ...)
    return obj.state_dict()


def extract_layer_info(state_dict: "OrderedDict[str, torch.Tensor]") -> list:
    """Return a list of dicts, one per parameter tensor."""
    layers = []
    for name, param in state_dict.items():
        data = param.detach().cpu().float().numpy()
        flat = data.flatten()
        layers.append(
            {
                "name": name,
                "shape": list(param.shape),
                "num_params": param.numel(),
                "dtype": str(param.dtype),
                "min": float(flat.min()),
                "max": float(flat.max()),
                "mean": float(flat.mean()),
                "std": float(flat.std()) if flat.size > 1 else 0.0,
                "l2_norm": float(np.linalg.norm(flat)),
                "data": data,
            }
        )
    return layers


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_and_save_architecture(state_dict, save_path: str = None) -> str:
    """Print a human-readable architecture table and optionally write it."""
    total_params = sum(p.numel() for p in state_dict.values())

    lines = []
    lines.append("=" * 80)
    lines.append("MODEL ARCHITECTURE")
    lines.append("=" * 80)

    current_module = ""
    for name, param in state_dict.items():
        parts = name.split(".")
        # Module = everything except the last token (weight / bias / etc.)
        module = ".".join(parts[:-1])
        leaf = parts[-1]

        if module != current_module:
            lines.append(f"\n[{module}]")
            current_module = module

        lines.append(f"  {leaf:<20s}  shape={str(list(param.shape)):<25s}  "
                     f"params={param.numel():>10,}")

    lines.append("\n" + "=" * 80)
    lines.append(f"Total parameters: {total_params:,}")
    lines.append("=" * 80)

    text = "\n".join(lines)
    print(text)

    if save_path:
        with open(save_path, "w") as fh:
            fh.write(text)

    return text


# ---------------------------------------------------------------------------
# Figure 1 – Summary dashboard
# ---------------------------------------------------------------------------

def plot_summary_dashboard(layers: list, save_path: str = None):
    """High-level overview: pie chart, global histogram, std bar, table."""
    total_params = sum(l["num_params"] for l in layers)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # ── Pie chart: params per top-level module ──────────────────────────────
    ax_pie = fig.add_subplot(gs[0, 0])
    module_params: dict = {}
    for layer in layers:
        top = layer["name"].split(".")[0]
        module_params[top] = module_params.get(top, 0) + layer["num_params"]
    sizes = list(module_params.values())
    labels = [f"{k}\n({v:,})" for k, v in module_params.items()]
    wedge_colors = plt.cm.tab20(np.linspace(0, 1, len(sizes)))
    ax_pie.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,
               colors=wedge_colors, textprops={"fontsize": 7})
    ax_pie.set_title("Parameter distribution\n(top-level modules)", fontsize=9)

    # ── Global weight histogram ─────────────────────────────────────────────
    ax_hist = fig.add_subplot(gs[0, 1])
    weight_layers = [l for l in layers if "weight" in l["name"].split(".")[-1]]
    if weight_layers:
        all_w = np.concatenate([l["data"].flatten() for l in weight_layers])
        p1, p99 = np.percentile(all_w, [1, 99])
        ax_hist.hist(np.clip(all_w, p1, p99), bins=120, color="steelblue",
                     alpha=0.85, density=True, edgecolor="none")
        ax_hist.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_hist.set_title("Global weight distribution\n(all weight tensors)", fontsize=9)
    ax_hist.set_xlabel("Weight value", fontsize=8)
    ax_hist.set_ylabel("Density", fontsize=8)

    # ── Per-layer std-dev bar chart ─────────────────────────────────────────
    ax_std = fig.add_subplot(gs[0, 2])
    if weight_layers:
        stds = [l["std"] for l in weight_layers]
        short_names = [
            ".".join(l["name"].split(".")[-2:]) for l in weight_layers
        ]
        bar_colors = plt.cm.plasma(np.array(stds) / (max(stds) or 1.0))
        ax_std.bar(range(len(stds)), stds, color=bar_colors)
        ax_std.set_xticks(range(len(short_names)))
        ax_std.set_xticklabels(short_names, rotation=90, fontsize=5)
        ax_std.set_ylabel("Std dev", fontsize=8)
    ax_std.set_title("Weight std-dev per layer", fontsize=9)

    # ── Layer statistics table ──────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis("off")

    headers = ["Layer", "Shape", "Params", "Min", "Max", "Mean", "Std"]
    rows = [
        [
            l["name"],
            str(l["shape"]),
            f"{l['num_params']:,}",
            f"{l['min']:.4f}",
            f"{l['max']:.4f}",
            f"{l['mean']:.4f}",
            f"{l['std']:.4f}",
        ]
        for l in layers
    ]
    max_rows = 22
    note = ""
    if len(rows) > max_rows:
        note = f"  (showing first {max_rows} of {len(layers)} layers)"
        rows = rows[:max_rows]

    tbl = ax_tbl.table(cellText=rows, colLabels=headers,
                       loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    tbl.scale(1, 1.35)
    ax_tbl.set_title(f"Layer statistics{note}", fontsize=9, pad=4)

    fig.suptitle(
        f"Model summary dashboard\nTotal parameters: {total_params:,}",
        fontsize=13, y=1.01,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 – Parameter counts per module
# ---------------------------------------------------------------------------

def plot_parameter_counts(layers: list, save_path: str = None):
    """Horizontal bar chart – parameter count grouped by 2-level module."""
    module_params: dict = {}
    for layer in layers:
        parts = layer["name"].split(".")
        # Use up to the first two path segments as the group key.
        key = ".".join(parts[: min(2, len(parts) - 1)]) or parts[0]
        module_params[key] = module_params.get(key, 0) + layer["num_params"]

    modules = list(module_params.keys())
    counts = list(module_params.values())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(modules)))

    fig, ax = plt.subplots(figsize=(12, max(5, len(modules) * 0.45 + 1)))
    bars = ax.barh(range(len(modules)), counts, color=colors, edgecolor="white",
                   linewidth=0.4)
    ax.set_yticks(range(len(modules)))
    ax.set_yticklabels(modules, fontsize=8)
    ax.set_xlabel("Number of parameters", fontsize=9)
    ax.set_title(
        f"Parameter count per module  (total: {sum(counts):,})", fontsize=10
    )

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() * 1.005,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center",
            fontsize=7,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 – Per-layer weight distributions
# ---------------------------------------------------------------------------

def plot_weight_distributions(layers: list, save_path: str = None,
                               max_panels: int = 28):
    """Grid of histograms, one per weight tensor (≥2-D)."""
    weight_layers = [
        l for l in layers
        if len(l["shape"]) >= 2 and "weight" in l["name"].split(".")[-1]
    ]

    if not weight_layers:
        return None

    if len(weight_layers) > max_panels:
        idxs = np.round(np.linspace(0, len(weight_layers) - 1,
                                     max_panels)).astype(int)
        weight_layers = [weight_layers[i] for i in idxs]

    ncols = 4
    nrows = (len(weight_layers) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4, nrows * 2.8))
    axes = np.array(axes).flatten()

    for i, layer in enumerate(weight_layers):
        ax = axes[i]
        flat = layer["data"].flatten()
        p1, p99 = np.percentile(flat, [1, 99])
        ax.hist(np.clip(flat, p1, p99), bins=60, color="steelblue",
                alpha=0.75, edgecolor="none")
        ax.axvline(0, color="crimson", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_title(layer["name"], fontsize=6.5, wrap=True)
        ax.set_xlabel("Value", fontsize=6)
        ax.tick_params(labelsize=5)
        stats = f"μ={layer['mean']:.3f}  σ={layer['std']:.3f}"
        ax.text(0.97, 0.96, stats, transform=ax.transAxes,
                fontsize=5.5, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Weight distributions per layer", fontsize=11, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 – 2-D weight heatmaps
# ---------------------------------------------------------------------------

def plot_weight_heatmaps(layers: list, save_path: str = None,
                          max_panels: int = 12):
    """Diverging RdBu heatmaps for 2-D weight matrices (size ≤ 512×512)."""
    candidates = [
        l for l in layers
        if len(l["shape"]) == 2
        and "weight" in l["name"].split(".")[-1]
        and l["shape"][0] <= 512
        and l["shape"][1] <= 512
    ]

    if not candidates:
        print("  No suitable 2-D weight matrices found for heatmaps – skipping.")
        return None

    if len(candidates) > max_panels:
        idxs = np.round(np.linspace(0, len(candidates) - 1,
                                     max_panels)).astype(int)
        candidates = [candidates[i] for i in idxs]

    ncols = min(3, len(candidates))
    nrows = (len(candidates) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5.5, nrows * 4.5))
    axes = np.array(axes).flatten()

    for i, layer in enumerate(candidates):
        ax = axes[i]
        data = layer["data"]
        vmax = float(np.percentile(np.abs(data), 99)) or 1e-6
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{layer['name']}\n{layer['shape']}", fontsize=7.5)
        ax.set_xlabel("Dim 1 →", fontsize=7)
        ax.set_ylabel("Dim 0 ↓", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Weight heatmaps (2-D layers)", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Figure 5 – L2 norms + parameter counts per layer
# ---------------------------------------------------------------------------

def plot_weight_norms(layers: list, save_path: str = None):
    """Two-row figure: L2 norms and parameter counts per individual layer."""
    names = [l["name"] for l in layers]
    norms = [l["l2_norm"] for l in layers]
    counts = [l["num_params"] for l in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1,
                                    figsize=(max(14, len(names) * 0.35), 10))

    # L2 norms ---------------------------------------------------------------
    norm_arr = np.array(norms)
    bar_colors = plt.cm.coolwarm(norm_arr / (norm_arr.max() or 1.0))
    ax1.bar(range(len(names)), norms, color=bar_colors, edgecolor="none")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=90, fontsize=5.5)
    ax1.set_ylabel("L2 norm", fontsize=9)
    ax1.set_title("Weight L2 norm per layer", fontsize=10)
    sm = ScalarMappable(cmap="coolwarm",
                        norm=Normalize(vmin=norm_arr.min(),
                                       vmax=norm_arr.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, orientation="vertical", fraction=0.02, pad=0.01)

    # Parameter counts -------------------------------------------------------
    cnt_arr = np.array(counts)
    bar_colors2 = plt.cm.viridis(cnt_arr / (cnt_arr.max() or 1.0))
    ax2.bar(range(len(names)), counts, color=bar_colors2, edgecolor="none")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=90, fontsize=5.5)
    ax2.set_ylabel("Number of parameters", fontsize=9)
    ax2.set_title("Parameter count per layer", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0 if sys.argv[1:] else 1)

    model_path = sys.argv[1]
    if not os.path.isfile(model_path):
        print(f"[error] model file not found: {model_path}")
        sys.exit(1)

    output_dir = (
        sys.argv[2] if len(sys.argv) > 2
        else os.path.dirname(os.path.abspath(model_path))
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint: {model_path}")
    state_dict = load_state_dict(model_path)

    print("Extracting layer information ...")
    layers = extract_layer_info(state_dict)
    total_params = sum(l["num_params"] for l in layers)
    print(f"  Layers   : {len(layers)}")
    print(f"  Params   : {total_params:,}")

    # Text summary -----------------------------------------------------------
    arch_path = os.path.join(output_dir, "architecture.txt")
    print_and_save_architecture(state_dict, save_path=arch_path)
    print(f"\nArchitecture written to: {arch_path}")

    # Figures ----------------------------------------------------------------
    plots = [
        ("Summary dashboard",            "summary_dashboard.png",
         lambda p: plot_summary_dashboard(layers, p)),
        ("Parameter counts per module",  "parameter_counts.png",
         lambda p: plot_parameter_counts(layers, p)),
        ("Weight distributions",         "weight_distributions.png",
         lambda p: plot_weight_distributions(layers, p)),
        ("Weight heatmaps",              "weight_heatmaps.png",
         lambda p: plot_weight_heatmaps(layers, p)),
        ("Weight L2 norms",              "weight_norms.png",
         lambda p: plot_weight_norms(layers, p)),
    ]

    print("\nGenerating visualisations ...")
    for idx, (label, fname, fn) in enumerate(plots, 1):
        fpath = os.path.join(output_dir, fname)
        print(f"  [{idx}/{len(plots)}] {label} ...")
        fig = fn(fpath)
        if fig is not None:
            plt.close(fig)

    print(f"\nAll outputs saved to: {output_dir}")
    print("Files generated:")
    for fname in ["architecture.txt"] + [p[1] for p in plots]:
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            print(f"  {fname:<35s}  {size:>10,} bytes")


if __name__ == "__main__":
    main()
